#!/usr/bin/env python3
"""
WFX (Gaussian XML-like wavefunction) to Gaussian Cube converter (pure Python)

- Input: Gaussian WFX file (XML-like). No third-party libraries used.
- Output: Gaussian .cube file with total electron density (bohr units).

Capabilities:
- Parses atoms, basis shells (s,p,d up to Cartesian), and MOs with occupancies.
- Expands shells to Cartesian AOs and evaluates density on a regular grid.
- Assumes basis is given as contracted shells with exponents and contraction
  coefficients (for each shell). Supports S, P, D (Cartesian) shells.

Limitations:
- Does not support higher angular momentum (f and above) yet.
- Assumes coordinates are in bohr. If your WFX stores angstrom, please convert or
  extend the parser to read unit tags.
- For speed and simplicity, density matrix is formed from MO coefficients and
  occupancies directly in Python; large systems may be slow.

Usage:
  python wfnx2cube.py -i input.wfx -o density.cube --spacing 0.2 --padding 3.0
"""

import argparse
import gzip
import math
import re
import sys
from typing import List, Tuple, Optional, Dict
import xml.etree.ElementTree as ET


# ----------------------------- Math utilities -----------------------------

def double_factorial(n: int) -> int:
    if n <= 0:
        return 1
    result = 1
    k = n
    while k > 1:
        result *= k
        k -= 2
    return result


def primitive_cart_normalization(alpha: float, lx: int, ly: int, lz: int) -> float:
    lsum = lx + ly + lz
    num = (2.0 ** (2 * lsum + 1.5)) * (alpha ** (lsum + 1.5))
    den = (math.pi ** 1.5) * double_factorial(2 * lx - 1) * double_factorial(2 * ly - 1) * double_factorial(2 * lz - 1)
    return math.sqrt(num / den)


# ----------------------------- Data structures -----------------------------

class Atom:
    def __init__(self, z: int, charge: float, x: float, y: float, zc: float):
        self.atomic_number = int(z)
        self.charge = float(charge)
        self.x = float(x)
        self.y = float(y)
        self.z = float(zc)

    @property
    def coord(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


class Shell:
    def __init__(self, center_index: int, L: int, exponents: List[float], coeffs: List[float]):
        self.center_index = int(center_index)  # 0-based
        self.L = int(L)  # 0=S, 1=P, 2=D (Cartesian)
        self.exponents = [float(x) for x in exponents]
        self.coeffs = [float(c) for c in coeffs]
        if len(self.exponents) != len(self.coeffs):
            raise ValueError("Exponents and coeffs length mismatch in shell")


class AOFunction:
    def __init__(self, center_index: int, lx: int, ly: int, lz: int, exponents: List[float], coeffs: List[float]):
        self.center_index = int(center_index)
        self.lx = int(lx)
        self.ly = int(ly)
        self.lz = int(lz)
        self.exponents = [float(x) for x in exponents]
        self.coeffs = [float(c) for c in coeffs]

    def value(self, r: Tuple[float, float, float], center: Tuple[float, float, float]) -> float:
        dx = r[0] - center[0]
        dy = r[1] - center[1]
        dz = r[2] - center[2]
        rr = dx * dx + dy * dy + dz * dz
        poly = (dx ** self.lx) * (dy ** self.ly) * (dz ** self.lz)
        total = 0.0
        for a, d in zip(self.exponents, self.coeffs):
            n = primitive_cart_normalization(a, self.lx, self.ly, self.lz)
            total += d * n * poly * math.exp(-a * rr)
        return total


class MolecularOrbital:
    def __init__(self, energy_au: float, occupancy: float, coeffs: List[float]):
        self.energy_au = float(energy_au)
        self.occupancy = float(occupancy)
        self.coeffs = [float(c) for c in coeffs]


class Wavefunction:
    def __init__(self, atoms: List[Atom], aos: List[AOFunction], mos: List[MolecularOrbital]):
        self.atoms = atoms
        self.aos = aos
        self.mos = mos
        if mos and len(aos) != len(mos[0].coeffs):
            raise ValueError("AO count does not match MO coefficient vector length")

    def build_density_ut(self, drop_threshold: float = 1e-14) -> List[Tuple[int, int, float]]:
        nao = len(self.aos)
        # Build upper triangle of D
        acc = {}
        for mo in self.mos:
            occ = mo.occupancy
            if abs(occ) < 1e-12:
                continue
            c = mo.coeffs
            for mu in range(nao):
                c_mu = c[mu]
                if c_mu == 0.0:
                    continue
                for nu in range(mu, nao):
                    c_nu = c[nu]
                    if c_nu == 0.0:
                        continue
                    acc[(mu, nu)] = acc.get((mu, nu), 0.0) + occ * c_mu * c_nu
        items = [(mu, nu, val) for (mu, nu), val in acc.items() if abs(val) > drop_threshold]
        items.sort(key=lambda t: (t[0], t[1]))
        return items


# ----------------------------- Parsing WFX -----------------------------

class WFXParseError(Exception):
    pass


def text_to_floats(text: str) -> List[float]:
    vals: List[float] = []
    for tok in text.replace('\n', ' ').replace('\r', ' ').replace('D', 'E').split():
        try:
            vals.append(float(tok))
        except ValueError:
            continue
    return vals


def parse_wfx(path: str) -> Wavefunction:
    # Load bytes, support gzip, strip leading noise, and legalize tag names with spaces
    try:
        with open(path, 'rb') as fb:
            raw = fb.read()
    except Exception as e:
        raise WFXParseError(f"Failed to read file: {e}")

    if len(raw) >= 2 and raw[0] == 0x1F and raw[1] == 0x8B:
        try:
            raw = gzip.decompress(raw)
        except Exception as e:
            raise WFXParseError(f"Failed to decompress gzip WFX: {e}")

    # Strip any bytes before first '<'
    lt = raw.find(b'<')
    if lt > 0:
        raw = raw[lt:]

    # Try parse as-is first
    text_variants: List[Tuple[str, str]] = []  # (label, text)
    for enc in ('utf-8', 'utf-16', 'utf-16le', 'latin-1'):
        try:
            text_variants.append((enc, raw.decode(enc)))
        except Exception:
            continue

    last_err = None
    root = None

    def legalize_tag_spaces(s: str) -> str:
        # Convert tags like <Number of Nuclei> to <NumberOfNuclei>
        tag_re = re.compile(r'<\s*/?\s*([^>]+?)\s*>')
        def repl(m: re.Match) -> str:
            inner = m.group(1)
            sin = inner.strip()
            if not sin:
                return m.group(0)
            if sin[0] in ('?', '!'):
                return m.group(0)
            closing = sin.startswith('/')
            if closing:
                sin = sin[1:].strip()
            # If attributes are present, leave unchanged
            if ('=' in sin) or ('"' in sin) or ("'" in sin):
                return m.group(0)
            name = sin.replace(' ', '')
            return f"<{('/' if closing else '')}{name}>"
        return tag_re.sub(repl, s)

    def norm_tag(tag: str) -> str:
        # Remove namespace and non-alnum, uppercase
        if '}' in tag:
            tag = tag.split('}', 1)[1]
        return ''.join(ch for ch in tag if ch.isalnum()).upper()

    def collect_texts(r: ET.Element) -> Dict[str, str]:
        d: Dict[str, str] = {}
        for el in r.iter():
            key = norm_tag(el.tag)
            txt = ''.join(el.itertext()) if el.text is not None else (''.join(el.itertext()) if list(el) else '')
            if txt is None:
                txt = ''
            d[key] = txt
        return d

    # Attempt parse with raw text variants; then with legalized tags
    for label, txt in text_variants:
        try:
            root = ET.fromstring(txt)
            tag_texts = collect_texts(root)
            break
        except Exception as e:
            last_err = e
            # Try legalized version
            try:
                fixed = legalize_tag_spaces(txt)
                root = ET.fromstring(fixed)
                tag_texts = collect_texts(root)
                break
            except Exception as e2:
                last_err = e2
                continue

    if root is None:
        raise WFXParseError(f"Failed to parse XML: {last_err}")

    def get_text_any(names: List[str]) -> Optional[str]:
        for nm in names:
            key = ''.join(ch for ch in nm if ch.isalnum()).upper()
            if key in tag_texts and tag_texts[key].strip():
                return tag_texts[key]
        return None

    # Atoms
    z_text = get_text_any(['AtomicNumbers', 'AtomZ', 'NuclearCharges', 'NuclearCharge'])
    if not z_text:
        raise WFXParseError("Missing atomic numbers/charges in WFX")
    zs = [int(round(v)) for v in text_to_floats(z_text)]

    coords_text = get_text_any(['NuclearCartesianCoordinates', 'NuclearCoordinates', 'AtomCartesianCoordinates'])
    if not coords_text:
        raise WFXParseError("Missing nuclear coordinates in WFX")
    coords = text_to_floats(coords_text)
    if len(coords) % 3 != 0:
        raise WFXParseError("Nuclear coordinates length is not a multiple of 3")
    if len(coords) // 3 != len(zs):
        raise WFXParseError("Mismatch between number of nuclei and coordinates")

    atoms: List[Atom] = []
    for i, Z in enumerate(zs):
        x, y, zc = coords[3 * i], coords[3 * i + 1], coords[3 * i + 2]
        atoms.append(Atom(Z, float(Z), x, y, zc))

    # Basis arrays
    centers_text = get_text_any(['ShellToAtomMap', 'ShellToNucleus', 'CenterIndex'])
    L_text = get_text_any(['ShellTypes', 'ShellAngularMomentum', 'AngularMomentum'])
    nprim_text = get_text_any(['NumberOfPrimitivesPerShell', 'NumberOfPrimitives'])
    exp_text = get_text_any(['PrimitiveGaussianExponents', 'PrimitiveExponents'])
    coef_text = get_text_any(['ContractionCoefficients', 'ContractionCoefficientsNormalized'])

    if not (centers_text and L_text and nprim_text and exp_text and coef_text):
        raise WFXParseError("Missing basis arrays (centers, L, nprims, exponents, coefficients)")

    centers = [int(round(v)) for v in text_to_floats(centers_text)]
    Ls = [int(round(v)) for v in text_to_floats(L_text)]
    nprims = [int(round(v)) for v in text_to_floats(nprim_text)]
    exps = text_to_floats(exp_text)
    coefs = text_to_floats(coef_text)

    if not (len(centers) == len(Ls) == len(nprims)):
        raise WFXParseError("Shell arrays length mismatch")

    shells: List[Shell] = []
    off = 0
    for ci, L, np in zip(centers, Ls, nprims):
        if L < 0:
            raise WFXParseError("SP shells (L<0) not supported in this version")
        if L > 2:
            raise WFXParseError("Only S/P/D shells supported in this version")
        part_exps = exps[off: off + np]
        part_coefs = coefs[off: off + np]
        if len(part_exps) != np or len(part_coefs) != np:
            raise WFXParseError("Primitive array slicing mismatch")
        shells.append(Shell(ci - 1, L, part_exps, part_coefs))
        off += np

    # Expand shells
    def expand_shell(shell: Shell) -> List[AOFunction]:
        aos: List[AOFunction] = []
        L = shell.L
        if L == 0:
            aos.append(AOFunction(shell.center_index, 0, 0, 0, shell.exponents, shell.coeffs))
        elif L == 1:
            aos.append(AOFunction(shell.center_index, 1, 0, 0, shell.exponents, shell.coeffs))
            aos.append(AOFunction(shell.center_index, 0, 1, 0, shell.exponents, shell.coeffs))
            aos.append(AOFunction(shell.center_index, 0, 0, 1, shell.exponents, shell.coeffs))
        elif L == 2:
            aos.append(AOFunction(shell.center_index, 2, 0, 0, shell.exponents, shell.coeffs))
            aos.append(AOFunction(shell.center_index, 0, 2, 0, shell.exponents, shell.coeffs))
            aos.append(AOFunction(shell.center_index, 0, 0, 2, shell.exponents, shell.coeffs))
            aos.append(AOFunction(shell.center_index, 1, 1, 0, shell.exponents, shell.coeffs))
            aos.append(AOFunction(shell.center_index, 1, 0, 1, shell.exponents, shell.coeffs))
            aos.append(AOFunction(shell.center_index, 0, 1, 1, shell.exponents, shell.coeffs))
        else:
            raise WFXParseError("Unsupported angular momentum L")
        return aos

    aos: List[AOFunction] = []
    for sh in shells:
        aos.extend(expand_shell(sh))

    # MOs
    occ_text = get_text_any(['OccupationNumbers', 'MOOccupation', 'Occupations', 'OccupancyNumbers'])
    if not occ_text:
        raise WFXParseError("Missing MO occupation numbers")
    occs = text_to_floats(occ_text)

    coeffs_text = get_text_any(['MOCoefficients', 'CoefficientMatrix'])
    mos: List[MolecularOrbital]
    if coeffs_text:
        flat = text_to_floats(coeffs_text)
        nao = len(aos)
        if nao == 0:
            raise WFXParseError("No AOs constructed from basis")
        if len(flat) % nao != 0:
            raise WFXParseError("Coefficient matrix length not divisible by AO count")
        nm = len(flat) // nao
        mos = []
        for i in range(nm):
            vec = flat[i * nao:(i + 1) * nao]
            occ = occs[i] if i < len(occs) else (2.0 if i * 2 < len(occs) else 0.0)
            mos.append(MolecularOrbital(0.0, occ, vec))
    else:
        # Try nested MO nodes, ignoring namespaces and name variations
        def is_mo_node(el: ET.Element) -> bool:
            t = norm_tag(el.tag)
            return t in ( 'MO', 'MOLECULARORBITAL', 'ORBITAL' )
        mo_nodes = [el for el in root.iter() if is_mo_node(el)]
        if not mo_nodes:
            raise WFXParseError("Missing MO coefficients")
        mos = []
        for i, mo in enumerate(mo_nodes):
            # Gather child texts
            child_map: Dict[str, str] = {}
            for ch in mo.iter():
                child_map[norm_tag(ch.tag)] = ''.join(ch.itertext()) if ch.text is not None else ''.join(ch.itertext())
            e_txt = child_map.get('ENERGY', '0.0')
            c_txt = child_map.get('COEFFICIENTS', '') or child_map.get('MOCOEFFICIENTS', '')
            coeffs = text_to_floats(c_txt)
            if len(coeffs) != len(aos):
                raise WFXParseError(f"MO {i+1} coefficient length {len(coeffs)} != AO count {len(aos)}")
            occ = occs[i] if i < len(occs) else (2.0 if i * 2 < len(occs) else 0.0)
            try:
                energy = float(e_txt.replace('D', 'E')) if e_txt else 0.0
            except Exception:
                energy = 0.0
            mos.append(MolecularOrbital(energy, occ, coeffs))

    return Wavefunction(atoms, aos, mos)


# ----------------------------- Grid and cube writer -----------------------------

def compute_bounding_box(atoms: List[Atom]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    xs = [a.x for a in atoms]
    ys = [a.y for a in atoms]
    zs = [a.z for a in atoms]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def compute_grid(atoms: List[Atom], spacing: float, padding: float) -> Tuple[Tuple[float, float, float], Tuple[int, int, int], Tuple[float, float, float]]:
    (xmin, ymin, zmin), (xmax, ymax, zmax) = compute_bounding_box(atoms)
    xmin -= padding
    ymin -= padding
    zmin -= padding
    xmax += padding
    ymax += padding
    zmax += padding
    nx = int(math.floor((xmax - xmin) / spacing)) + 1
    ny = int(math.floor((ymax - ymin) / spacing)) + 1
    nz = int(math.floor((zmax - zmin) / spacing)) + 1
    return (xmin, ymin, zmin), (nx, ny, nz), (spacing, spacing, spacing)


def write_cube_density(output_path: str,
                       atoms: List[Atom],
                       aos: List[AOFunction],
                       D_ut: List[Tuple[int, int, float]],
                       origin: Tuple[float, float, float],
                       grid_shape: Tuple[int, int, int],
                       spacing: Tuple[float, float, float],
                       max_points: Optional[int] = None,
                       progress: bool = True) -> None:
    nx, ny, nz = grid_shape
    total_points = nx * ny * nz
    if max_points is not None and total_points > max_points:
        raise RuntimeError(f"Grid too large: {total_points} > max_points {max_points}")

    with open(output_path, 'w', encoding='utf-8') as fo:
        fo.write("WFX to cube (density)\n")
        fo.write("Generated by wfnx2cube.py (units: bohr)\n")
        fo.write(f"{len(atoms):5d} {origin[0]:14.6f} {origin[1]:14.6f} {origin[2]:14.6f}\n")
        fo.write(f"{nx:5d} {spacing[0]:14.6f} {0.0:14.6f} {0.0:14.6f}\n")
        fo.write(f"{ny:5d} {0.0:14.6f} {spacing[1]:14.6f} {0.0:14.6f}\n")
        fo.write(f"{nz:5d} {0.0:14.6f} {0.0:14.6f} {spacing[2]:14.6f}\n")
        for a in atoms:
            fo.write(f"{a.atomic_number:5d} {a.charge:14.6f} {a.x:14.6f} {a.y:14.6f} {a.z:14.6f}\n")

        x0, y0, z0 = origin
        dx, dy, dz = spacing
        nao = len(aos)
        buf: List[str] = []
        nline = 0
        printed = False

        for ix in range(nx):
            x = x0 + ix * dx
            if progress and nx > 1 and ix % max(1, nx // 20) == 0:
                pct = 100.0 * ix / max(1, nx - 1)
                print(f"Progress: {pct:5.1f}% (ix={ix}/{nx})", file=sys.stderr)
                printed = True
            for iy in range(ny):
                y = y0 + iy * dy
                for iz in range(nz):
                    z = z0 + iz * dz
                    r = (x, y, z)
                    ao_vals = [0.0] * nao
                    for mu in range(nao):
                        ao_vals[mu] = aos[mu].value(r, atoms[aos[mu].center_index].coord)
                    rho = 0.0
                    for (mu, nu, d) in D_ut:
                        if mu == nu:
                            rho += d * ao_vals[mu] * ao_vals[nu]
                        else:
                            rho += 2.0 * d * ao_vals[mu] * ao_vals[nu]
                    buf.append(f"{rho:13.5e}")
                    nline += 1
                    if nline == 6:
                        fo.write(" ".join(buf) + "\n")
                        buf.clear()
                        nline = 0
        if nline:
            fo.write(" ".join(buf) + "\n")
        if progress and printed:
            print("Progress: 100.0%", file=sys.stderr)


# ----------------------------- CLI -----------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Convert Gaussian WFX (XML) to Gaussian cube (electron density). Units: bohr")
    p.add_argument('-i', '--input', required=True, help='Input WFX file path')
    p.add_argument('-o', '--output', required=True, help='Output cube file path')
    p.add_argument('--spacing', type=float, default=0.2, help='Grid spacing (bohr)')
    p.add_argument('--padding', type=float, default=3.0, help='Padding around molecule (bohr)')
    p.add_argument('--d-thresh', type=float, default=1e-12, help='Drop |D(mu,nu)| below threshold')
    p.add_argument('--max-points', type=int, default=6_000_000, help='Abort if grid points exceed this')
    p.add_argument('--quiet', action='store_true', help='Reduce progress output')

    args = p.parse_args(argv)

    try:
        wf = parse_wfx(args.input)
    except WFXParseError as e:
        print(f"WFX parse error: {e}", file=sys.stderr)
        return 2

    origin, shape, spacing_vec = compute_grid(wf.atoms, args.spacing, args.padding)
    if not args.quiet:
        nx, ny, nz = shape
        print(f"Atoms: {len(wf.atoms)}  AOs: {len(wf.aos)}  MOs: {len(wf.mos)}", file=sys.stderr)
        print(f"Grid: {nx} x {ny} x {nz} (~{nx*ny*nz:,} points)", file=sys.stderr)

    D_ut = wf.build_density_ut(drop_threshold=args.d_thresh)
    if not args.quiet:
        print(f"Density matrix nnz (upper): {len(D_ut)}", file=sys.stderr)

    try:
        write_cube_density(
            output_path=args.output,
            atoms=wf.atoms,
            aos=wf.aos,
            D_ut=D_ut,
            origin=origin,
            grid_shape=shape,
            spacing=spacing_vec,
            max_points=args.max_points,
            progress=(not args.quiet),
        )
    except Exception as e:
        print(f"Failed writing cube: {e}", file=sys.stderr)
        return 3

    if not args.quiet:
        print(f"Wrote cube to: {args.output}", file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())

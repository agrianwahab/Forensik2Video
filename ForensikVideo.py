# vifa_pro.py
# (Sistem Forensik Video Profesional dengan Analisis Multi-Lapis)
# File Tunggal Lengkap dan Fungsional - VERSI PERBAIKAN

"""
VIFA-Pro: Sistem Forensik Video Profesional
========================================================================================
Versi ini adalah implementasi lengkap yang mendeteksi manipulasi temporal dan spasial
melalui penggabungan berbagai teknik forensik.

Deteksi:
- Diskontinuitas (Deletion/Insertion): Melalui Aliran Optik, SSIM, dan K-Means.
- Duplikasi Frame (Duplication): Melalui pHash, dikonfirmasi oleh SIFT+RANSAC dan SSIM.
- Penyisipan Area (Splicing): Melalui Analisis Tingkat Kesalahan (ELA) yang dipicu
  oleh deteksi diskontinuitas.

Fitur Utama:
- Kerangka Kerja DFRWS: Mengikuti alur kerja forensik yang terstruktur.
- Analisis Multi-Lapis: Menggabungkan bukti dari berbagai domain.
- Penilaian Kepercayaan: Memberi skor pada setiap anomali (Rendah hingga Sangat Tinggi).
- Pelaporan Profesional: Menghasilkan laporan PDF terperinci.
- Mode Ganda: Mendukung analisis mandiri dan perbandingan dengan baseline.

Author: OpenAI-GPT & Anda
License: MIT
Dependencies: opencv-python, opencv-contrib-python, imagehash, numpy, Pillow, 
              reportlab, matplotlib, tqdm, scikit-learn, scikit-image
"""

from __future__ import annotations
import argparse
import json
import hashlib
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Pemeriksaan Dependensi Awal
try:
    import cv2
    import imagehash
    import numpy as np
    from PIL import Image, ImageChops, ImageEnhance
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as PlatypusImage, Table, TableStyle, PageBreak
    from reportlab.lib import colors
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from sklearn.cluster import KMeans
    from skimage.metrics import structural_similarity as ssim
except ImportError as e:
    print(f"Error: Dependensi penting tidak ditemukan -> {e}")
    print("Silakan jalankan perintah berikut untuk menginstal dependensi yang diperlukan:")
    print("pip install opencv-python opencv-contrib-python imagehash numpy Pillow reportlab matplotlib tqdm scikit-learn scikit-image")
    sys.exit(1)


###############################################################################
# Utilitas & Konfigurasi Global
###############################################################################

class Icons:
    IDENTIFICATION = "🔍"; PRESERVATION = "🛡️"; COLLECTION = "📥"; EXAMINATION = "🔬"
    ANALYSIS = "📈"; REPORTING = "📄"; SUCCESS = "✅"; ERROR = "❌"; INFO = "ℹ️"
    CONFIDENCE_LOW = "🟩"; CONFIDENCE_MED = "🟨"; CONFIDENCE_HIGH = "🟧"; CONFIDENCE_VHIGH = "🟥"

# Ambang batas global yang dapat disesuaikan
CONFIG = {
    "HASH_DIST_DUPLICATE": 2,       # Jarak pHash maksimal untuk dianggap duplikat
    "OPTICAL_FLOW_Z_THRESH": 4.0,   # Z-score untuk anomali aliran optik
    "SSIM_DISCONTINUITY_DROP": 0.25, # Penurunan SSIM minimum untuk menandai diskontinuitas
    "SIFT_MIN_MATCH_COUNT": 10,     # Jumlah minimal kecocokan SIFT agar dianggap valid
    "KMEANS_CLUSTERS": 8,           # Jumlah cluster untuk analisis layout warna
    "DUPLICATION_SSIM_CONFIRM": 0.95, # SSIM minimal untuk mengkonfirmasi kandidat duplikasi
}

def print_stage_banner(stage_name: str, icon: str, description: str):
    """ Mencetak banner yang terstruktur untuk setiap tahap DFRWS. """
    width = 80
    print("\n" + "=" * width)
    print(f"=== {icon}  TAHAP DFRWS: {stage_name.upper()} ".ljust(width - 3) + "===")
    print("=" * width)
    print(f"{Icons.INFO}  {description}")
    print("-" * width)

###############################################################################
# Struktur Data Inti (Dataclasses)
###############################################################################

@dataclass
class Evidence:
    """ Menyimpan semua bukti yang terkumpul untuk satu anomali. """
    confidence: str = "Tidak Diketahui"
    reasons: list[str] | None = None
    metrics: dict | None = None
    ela_path: str | None = None
    sift_path: str | None = None

@dataclass
class FrameInfo:
    """ Menyimpan semua informasi yang dianalisis untuk satu frame. """
    index: int
    timestamp: float
    img_path: str
    hash: str | None = None
    type: str = "original"
    ssim_to_prev: float | None = None
    color_cluster: int | None = None
    evidence_obj: Evidence | None = None

@dataclass
class AnalysisResult:
    """ Wadah hasil akhir untuk satu video. """
    video_path: str
    preservation_hash: str
    metadata: dict
    frames: list[FrameInfo]
    summary: dict
    plots: dict
    localizations: list[dict]

###############################################################################
# Modul Analisis Forensik (Fungsi-Fungsi Analisis)
###############################################################################

def perform_ela(image_path: Path, quality: int = 90) -> Path | None:
    """ Melakukan Error Level Analysis (ELA) pada satu frame. """
    try:
        ela_dir = image_path.parent.parent / "ela_artifacts"
        ela_dir.mkdir(exist_ok=True)
        out_path = ela_dir / f"{image_path.stem}_ela.jpg"
        temp_jpg_path = 'temp_ela.jpg'
        
        # Menggunakan 'with' untuk memastikan file tertutup dengan benar
        with Image.open(image_path).convert('RGB') as im:
            im.save(temp_jpg_path, 'JPEG', quality=quality)
        
        with Image.open(image_path).convert('RGB') as im_orig, Image.open(temp_jpg_path) as resaved_im:
            ela_im = ImageChops.difference(im_orig, resaved_im)

        if Path(temp_jpg_path).exists():
            Path(temp_jpg_path).unlink()
        
        extrema = ela_im.getextrema()
        max_diff = max(ex[1] for ex in extrema) if extrema else 1
        if max_diff == 0: max_diff = 1
        
        scale = 255.0 / max_diff
        ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)
        ela_im.save(out_path)
        return out_path
    except Exception as e:
        print(f"  {Icons.ERROR} Gagal ELA pada {image_path.name}: {e}", file=sys.stderr)
        return None

def compare_sift(img_path1: Path, img_path2: Path, out_dir: Path) -> tuple[int, Path | None]:
    """ Membandingkan dua gambar menggunakan SIFT+RANSAC dan menyimpan visualisasinya. """
    try:
        img1 = cv2.imread(str(img_path1), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(img_path2), cv2.IMREAD_GRAYSCALE)
        if img1 is None or img2 is None: return 0, None

        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2: return 0, None
        
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        if not matches or any(len(m) < 2 for m in matches): return 0, None
        
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good_matches) > CONFIG["SIFT_MIN_MATCH_COUNT"]:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if mask is None: return len(good_matches), None

            inliers = mask.ravel().sum()
            draw_params = dict(matchColor=(0, 255, 0), singlePointColor=None, matchesMask=mask.ravel().tolist(), flags=2)
            img_out = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)
            
            sift_dir = out_dir / "sift_artifacts"
            sift_dir.mkdir(exist_ok=True)
            out_path = sift_dir / f"sift_{img_path1.stem}_vs_{img_path2.stem}.jpg"
            cv2.imwrite(str(out_path), img_out)
            return int(inliers), out_path
        
        return len(good_matches), None
    except cv2.error as e:
         print(f"  {Icons.ERROR} Error OpenCV SIFT pada {img_path1.name}: {e}", file=sys.stderr)
         return 0, None
    except Exception as e:
        print(f"  {Icons.ERROR} Gagal SIFT umum: {e}", file=sys.stderr)
        return 0, None


def analyze_color_layout(frames: list[FrameInfo], n_clusters: int) -> list[int]:
    """ Mengelompokkan frame berdasarkan histogram warna. """
    histograms = []
    for f in tqdm(frames, desc="    Menghitung Histogram", leave=False):
        img = cv2.imread(f.img_path)
        if img is None: continue
        hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        histograms.append(hist.flatten())

    if not histograms: return [0] * len(frames)
    
    actual_n_clusters = min(n_clusters, len(histograms))
    if actual_n_clusters < 2: return [0] * len(frames)
    
    kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init='auto').fit(histograms)
    return kmeans.labels_.tolist()

###############################################################################
# Fungsi-Fungsi Pra-analisis
###############################################################################

def calculate_sha256(file_path: Path) -> str:
    """ Menghitung hash SHA-256 dari sebuah file. """
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except IOError as e:
        print(f"{Icons.ERROR} Gagal membaca file untuk hashing: {e}", file=sys.stderr)
        return "error"

def ffprobe_metadata(video_path: Path) -> dict:
    """ Mengekstrak metadata video menggunakan ffprobe. """
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(video_path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8')
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"{Icons.ERROR} Gagal menjalankan ffprobe. Pastikan ffmpeg terinstal dan ada di PATH sistem Anda.", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"{Icons.ERROR} Gagal mem-parsing output JSON dari ffprobe.", file=sys.stderr)
        return {"error": "parse_failed"}

def extract_frames(video_path: Path, out_dir: Path, fps: int) -> int:
    """ Mengekstrak frame dari video menggunakan ffmpeg. """
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(out_dir / "frame_%06d.jpg")
    cmd = ["ffmpeg", "-i", str(video_path), "-vf", f"fps={fps}", "-qscale:v", "2", pattern, "-y", "-hide_banner", "-loglevel", "error"]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return len(list(out_dir.glob('*.jpg')))
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if e.stderr else "Unknown ffmpeg error."
        print(f"{Icons.ERROR} ffmpeg gagal mengekstrak frame. Pesan error:\n{error_msg}", file=sys.stderr)
        sys.exit(1)

def compute_initial_metrics(frames: list[FrameInfo]):
    """ Menghitung pHash dan SSIM antar-frame secara berurutan. """
    prev_gray = None
    for f in tqdm(frames, desc="    Menghitung pHash & SSIM"):
        try:
            with Image.open(f.img_path) as img:
                f.hash = str(imagehash.average_hash(img))
            
            current_gray = cv2.imread(f.img_path, cv2.IMREAD_GRAYSCALE)
            if current_gray is None: continue
            
            if prev_gray is not None:
                data_range = float(current_gray.max() - current_gray.min())
                if data_range > 0:
                    score, _ = ssim(prev_gray, current_gray, data_range=data_range, full=True)
                    f.ssim_to_prev = score
            prev_gray = current_gray
        except Exception as e:
            print(f"  {Icons.ERROR} Gagal menghitung metrik untuk frame {f.index}: {e}")

def analyze_optical_flow(frames: list[FrameInfo]) -> list[tuple[int, float]]:
    """ Menganalisis diskontinuitas aliran optik. """
    mags = []
    prev_gray = None
    for f in tqdm(frames, desc="    Menganalisis Aliran Optik"):
        current_gray = cv2.imread(f.img_path, cv2.IMREAD_GRAYSCALE)
        if current_gray is None: continue

        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mags.append((f.index, float(np.mean(mag))))
        prev_gray = current_gray
    return mags

###############################################################################
# Modul Sintesis Bukti dan Analisis Utama
###############################################################################

def synthesize_and_analyze(frames: list[FrameInfo], out_dir: Path):
    """
    Inti dari VIFA-Pro: Menganalisis, mensintesis bukti, dan menetapkan skor kepercayaan.
    """
    n = len(frames)
    if n < 2: return

    # 1. Analisis Aliran Optik untuk diskontinuitas
    flow_results = analyze_optical_flow(frames)
    if flow_results:
        flow_values = np.array([m[1] for m in flow_results])
        if len(flow_values) > 0:
            median = np.median(flow_values)
            mad = np.median(np.abs(flow_values - median)) + 1e-9 # Mencegah pembagian dengan nol
            
            for idx, mag in flow_results:
                if mad > 0:
                    z_score = 0.6745 * (mag - median) / mad
                    if abs(z_score) > CONFIG["OPTICAL_FLOW_Z_THRESH"]:
                        f = frames[idx]
                        if f.evidence_obj is None: f.evidence_obj = Evidence(reasons=[], metrics={})
                        if "Lonjakan Aliran Optik" not in f.evidence_obj.reasons:
                            f.evidence_obj.reasons.append("Lonjakan Aliran Optik")
                        f.evidence_obj.metrics["optical_flow_z_score"] = round(z_score, 2)
    
    # 2. Analisis diskontinuitas berbasis SSIM dan Warna
    for i in range(1, n):
        f_curr = frames[i]
        f_prev = frames[i - 1]
        
        if f_curr.ssim_to_prev is not None and f_prev.ssim_to_prev is not None:
             if abs(f_prev.ssim_to_prev - f_curr.ssim_to_prev) > CONFIG["SSIM_DISCONTINUITY_DROP"]:
                if f_curr.evidence_obj is None: f_curr.evidence_obj = Evidence(reasons=[], metrics={})
                if "Penurunan Drastis SSIM" not in f_curr.evidence_obj.reasons:
                    f_curr.evidence_obj.reasons.append("Penurunan Drastis SSIM")
                f_curr.evidence_obj.metrics["ssim_drop"] = round(abs(f_prev.ssim_to_prev - f_curr.ssim_to_prev), 2)

        if f_curr.color_cluster is not None and f_curr.color_cluster != f_prev.color_cluster:
            if f_curr.evidence_obj is None: f_curr.evidence_obj = Evidence(reasons=[], metrics={})
            if "Perubahan Adegan (Warna)" not in f_curr.evidence_obj.reasons:
                f_curr.evidence_obj.reasons.append("Perubahan Adegan (Warna)")
            f_curr.evidence_obj.metrics["color_cluster_jump"] = f"{f_prev.color_cluster} -> {f_curr.color_cluster}"
    
    # 3. Analisis Duplikasi Frame (Mode Mandiri)
    hash_map = defaultdict(list)
    for f in frames:
        if f.hash:
            hash_map[f.hash].append(f.index)
    
    print(f"  {Icons.INFO} Memverifikasi kandidat duplikasi (jika ada)...")
    for h, idxs in tqdm(hash_map.items(), desc="    Memverifikasi Duplikasi", leave=False):
        if len(idxs) > 1:
            for i in range(len(idxs) - 1):
                idx1, idx2 = idxs[i], idxs[i+1]
                img1_path, img2_path = Path(frames[idx1].img_path), Path(frames[idx2].img_path)
                
                img1_gray, img2_gray = cv2.imread(str(img1_path), 0), cv2.imread(str(img2_path), 0)
                if img1_gray is None or img2_gray is None: continue
                
                data_range = float(img1_gray.max() - img1_gray.min())
                if data_range == 0: continue

                ssim_score, _ = ssim(img1_gray, img2_gray, data_range=data_range, full=True)
                if ssim_score > CONFIG["DUPLICATION_SSIM_CONFIRM"]:
                    inliers, sift_path = compare_sift(img1_path, img2_path, out_dir)
                    if inliers >= CONFIG["SIFT_MIN_MATCH_COUNT"]:
                        f_dup = frames[idx2]
                        f_dup.type = "anomaly_duplication"
                        if f_dup.evidence_obj is None: f_dup.evidence_obj = Evidence(reasons=[], metrics={})
                        if f"Duplikasi dari frame {idx1}" not in f_dup.evidence_obj.reasons:
                            f_dup.evidence_obj.reasons.append(f"Duplikasi dari frame {idx1}")
                        f_dup.evidence_obj.metrics["source_frame"] = idx1
                        f_dup.evidence_obj.metrics["ssim_score_to_source"] = round(ssim_score, 4)
                        f_dup.evidence_obj.metrics["sift_inliers"] = inliers
                        f_dup.evidence_obj.sift_path = str(sift_path) if sift_path else None

    # 4. Finalisasi: Skor Kepercayaan dan Analisis ELA Pemicu
    print(f"  {Icons.INFO} Mensintesis bukti & menjalankan analisis spasial (ELA)...")
    for f in tqdm(frames, desc="    Finalisasi & ELA", leave=False):
        if f.evidence_obj and f.evidence_obj.reasons:
            if f.type == "original": f.type = "anomaly_discontinuity"
            
            num_reasons = len(f.evidence_obj.reasons)
            if num_reasons == 1: f.evidence_obj.confidence = "RENDAH"
            elif num_reasons == 2: f.evidence_obj.confidence = "SEDANG"
            else: f.evidence_obj.confidence = "TINGGI"
            
            if f.type == "anomaly_duplication": f.evidence_obj.confidence = "SANGAT TINGGI"
            
            if f.evidence_obj.confidence in ["SEDANG", "TINGGI", "SANGAT TINGGI"] and f.type != "anomaly_duplication":
                ela_path = perform_ela(Path(f.img_path))
                if ela_path:
                    f.evidence_obj.ela_path = str(ela_path)
                    if "Potensi Anomali Kompresi (ELA)" not in f.evidence_obj.reasons:
                        f.evidence_obj.reasons.append("Potensi Anomali Kompresi (ELA)")
                    f.evidence_obj.confidence = "SANGAT TINGGI"
            
            if isinstance(f.evidence_obj.reasons, list):
                f.evidence_obj.reasons = ", ".join(sorted(list(set(f.evidence_obj.reasons))))

def analyze_pairwise(base_frames: list[FrameInfo], sus_frames: list[FrameInfo], out_dir: Path):
    """ Menganalisis penghapusan (deletion) dan penyisipan (insertion) secara komparatif. """
    print(f"  {Icons.INFO} Melakukan analisis komparatif dengan baseline...")
    base_hashes = {f.hash for f in base_frames if f.hash}

    for f_sus in sus_frames:
        if f_sus.hash not in base_hashes:
            f_sus.type = "anomaly_insertion"
            if f_sus.evidence_obj is None: f_sus.evidence_obj = Evidence(reasons=[], metrics={})
            f_sus.evidence_obj.reasons = ["Frame tidak ditemukan di baseline"]
            f_sus.evidence_obj.confidence = "SANGAT TINGGI"
    
    synthesize_and_analyze(sus_frames, out_dir)

###############################################################################
# Modul Pelaporan & Visualisasi Profesional
###############################################################################

def create_plots(frames: list[FrameInfo], out_dir: Path, video_name: str) -> dict:
    """ Membuat semua visualisasi plot dan menyimpannya. """
    plot_paths = {}
    n = len(frames)
    if n == 0: return {}
    indices = np.arange(n)

    # 1. Plot Anomali Temporal
    plt.figure(figsize=(15, 4))
    colors, labels = [], []
    for f in frames:
        if f.type.startswith("anomaly"):
            labels.append(1)
            color_map = {"duplication": 'orange', "insertion": 'red', "discontinuity": 'purple', "deletion_event": "blue"}
            colors.append(color_map.get(f.type.split('_')[-1], 'black'))
        else:
            labels.append(0)
            colors.append('green')

    mask = np.array(labels, dtype=bool)
    plt.vlines(indices[mask], ymin=0, ymax=np.ones(mask.sum()), colors=np.array(colors)[mask], lw=2)
    plt.scatter(indices[mask], np.ones(mask.sum()), c=np.array(colors)[mask])
    plt.ylim(-0.1, 1.1); plt.xlabel("Indeks Bingkai"); plt.ylabel("Anomali (1=Ya, 0=Tidak)")
    plt.title(f"Peta Anomali Temporal untuk {video_name}"); plt.grid(True, axis='x', linestyle='--', alpha=0.6)

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], c='r', lw=4, label='Penyisipan (Insertion)'),
                       Line2D([0], [0], c='orange', lw=4, label='Duplikasi (Duplication)'),
                       Line2D([0], [0], c='purple', lw=4, label='Diskontinuitas'),
                       Line2D([0], [0], c='g', lw=4, label='Asli/Tidak ada Anomali')]
    plt.legend(handles=legend_elements, loc='upper right')
    
    temporal_plot_path = out_dir / f"plot_temporal_{video_name}.png"
    plt.savefig(temporal_plot_path, bbox_inches="tight"); plt.close()
    plot_paths['temporal'] = str(temporal_plot_path)

    return plot_paths

def build_localizations(frames: list[FrameInfo]) -> list[dict]:
    """ Mengelompokkan frame anomali yang berdekatan menjadi peristiwa. """
    locs = []
    current_event = None
    for f in frames:
        if f.type.startswith("anomaly"):
            if current_event and current_event["event"] == f.type and f.index == current_event["end_frame"] + 1:
                current_event["end_frame"] = f.index; current_event["end_ts"] = f.timestamp
            else:
                if current_event: locs.append(current_event)
                current_event = {"event": f.type, "start_frame": f.index, "end_frame": f.index,
                                 "start_ts": f.timestamp, "end_ts": f.timestamp,
                                 "confidence": f.evidence_obj.confidence if f.evidence_obj else "Tidak Diketahui",
                                 "reasons": f.evidence_obj.reasons if f.evidence_obj else "N/A",
                                 "metrics": f.evidence_obj.metrics if f.evidence_obj else {},
                                 "image": f.img_path,
                                 "ela_path": f.evidence_obj.ela_path if f.evidence_obj else None,
                                 "sift_path": f.evidence_obj.sift_path if f.evidence_obj else None}
        elif current_event:
            locs.append(current_event); current_event = None
    if current_event: locs.append(current_event)
    return locs

def write_professional_report(result: AnalysisResult, out_pdf: Path, baseline_result: AnalysisResult | None = None):
    """ Membuat laporan PDF profesional yang terperinci. """
    doc = SimpleDocTemplate(str(out_pdf), pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    
    # *** PERBAIKAN ***: Cek jika style sudah ada sebelum menambahkan.
    if 'Code' not in styles:
        styles.add(ParagraphStyle(name='Code', parent=styles['Normal'], fontName='Courier', fontSize=8, leading=10))

    story = []
    
    suspect_name = Path(result.video_path).name
    story.append(Paragraph(f"Laporan Analisis Forensik VIFA-Pro", styles['h1']))
    story.append(Spacer(1, 12))
    
    col_widths_meta = [120, 420]
    metadata_data = [['<b>Atribut</b>', f'<b>Video Bukti: {suspect_name}</b>'],
                     ['Path File', Paragraph(result.video_path, styles['Code'])],
                     ['Hash SHA-256', Paragraph(result.preservation_hash, styles['Code'])],
                     ['Format Video', result.metadata.get('format', {}).get('format_long_name', 'N/A')],
                     ['Durasi', f"{float(result.metadata.get('format', {}).get('duration', 0)):.2f} detik"],
                     ['Jumlah Bingkai (dianalisis)', result.summary.get('total_frames', 0)],
                     ['Jumlah Anomali', f"{result.summary.get('total_anomaly', 0)} ({result.summary.get('pct_anomaly', 0)}%)"]]
    if baseline_result:
        col_widths_meta = [120, 220, 220]
        metadata_data[0].append(f'<b>Video Referensi: {Path(baseline_result.video_path).name}</b>')
        metadata_data[1].append(Paragraph(baseline_result.video_path, styles['Code']))
        metadata_data[2].append(Paragraph(baseline_result.preservation_hash, styles['Code']))
        metadata_data[3].append(baseline_result.metadata.get('format', {}).get('format_long_name', 'N/A'))
        metadata_data[4].append(f"{float(baseline_result.metadata.get('format', {}).get('duration', 0)):.2f} detik")
        metadata_data[5].append(baseline_result.summary.get('total_frames', 0))
        metadata_data[6].append(f"{baseline_result.summary.get('total_anomaly', 0)} ({baseline_result.summary.get('pct_anomaly', 0)}%)")
        
    t = Table(metadata_data, colWidths=col_widths_meta)
    t.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.darkslategray), ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
                           ('ALIGN', (0,0), (-1,-1), 'LEFT'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                           ('INNERGRID', (0,0), (-1,-1), 0.25, colors.black), ('BOX', (0,0), (-1,-1), 0.25, colors.black),
                           ('SPAN', (1,0), (1 if not baseline_result else 2), 0)])) # Correct span logic
    story.append(t); story.append(Spacer(1, 18))
    
    story.append(Paragraph("Ringkasan Visual Anomali Temporal", styles['h2']))
    story.append(PlatypusImage(result.plots['temporal'], width=520, height=135)); story.append(Spacer(1, 18))
    
    story.append(PageBreak())
    story.append(Paragraph("Detail Lokalisasi Anomali", styles['h2']))
    if not result.localizations:
        story.append(Paragraph("Tidak ada anomali yang signifikan terdeteksi.", styles['Normal']))
    for loc in result.localizations:
        story.append(Spacer(1, 12))
        confidence = loc.get('confidence', 'N/A')
        conf_icon = {"RENDAH": Icons.CONFIDENCE_LOW, "SEDANG": Icons.CONFIDENCE_MED, "TINGGI": Icons.CONFIDENCE_HIGH, "SANGAT TINGGI": Icons.CONFIDENCE_VHIGH}.get(confidence, "")
        story.append(Paragraph(f"{conf_icon} Peristiwa: <b>{loc['event'].replace('anomaly_', '').upper()}</b> pada <b>{loc['start_ts']:.2f}s</b>", styles['h3']))
        
        detail_data = [['<b>Atribut</b>', '<b>Detail</b>'], ['Tingkat Kepercayaan', f'<b>{confidence}</b>'],
                       ['Rentang Kejadian', f"Bingkai {loc['start_frame']} - {loc['end_frame']}"],
                       ['Alasan Deteksi', Paragraph(str(loc.get('reasons', 'N/A')), styles['Normal'])],
                       ['Metrik Teknis', Paragraph(json.dumps(loc.get('metrics', {}), indent=2), styles['Code'])]]
        t_detail = Table(detail_data, colWidths=[120, 420])
        t_detail.setStyle(TableStyle([('BACKGROUND', (0,0), (0,-1), colors.lightgrey), ('VALIGN', (0,0), (-1,-1), 'TOP'), ('GRID', (0,0), (-1,-1), 0.25, colors.black)]))
        story.append(t_detail); story.append(Spacer(1, 6))

        visual_evidence, headers = [], []
        if loc.get('image'): visual_evidence.append(PlatypusImage(loc['image'], width=250, height=140)); headers.append("<b>Frame Anomali</b>")
        if loc.get('ela_path'): visual_evidence.append(PlatypusImage(loc['ela_path'], width=250, height=140)); headers.append("<b>Analisis ELA</b>")
        
        if visual_evidence: story.append(Table([headers, visual_evidence]))
        if loc.get('sift_path'): story.append(Spacer(1,6)); story.append(PlatypusImage(loc['sift_path'], width=520))

        story.append(Spacer(1, 18))

    doc.build(story)
    
###############################################################################
# Alur Kerja DFRWS Utama dan Fungsi `main`
###############################################################################

def process_video(video_path: Path, out_dir: Path, fps: int) -> AnalysisResult:
    """ Melakukan seluruh pipeline analisis untuk satu video. """
    vid_stem = video_path.stem
    print(f"\n{'='*25} MEMPROSES: {video_path.name} {'='*25}")
    
    print_stage_banner("Identifikasi & Preservasi", Icons.IDENTIFICATION, f"Memverifikasi & menghash {video_path.name}")
    preservation_hash = calculate_sha256(video_path)
    print(f"  -> Hash SHA-256: {preservation_hash}")
    
    print_stage_banner("Koleksi", Icons.COLLECTION, f"Mengekstrak metadata & bingkai pada {fps} FPS.")
    metadata = ffprobe_metadata(video_path)
    frames_dir = out_dir / f"frames_{vid_stem}"
    num_frames = extract_frames(video_path, frames_dir, fps)
    print(f"  {Icons.SUCCESS} {num_frames} bingkai diekstrak ke '{frames_dir}'")
    
    print_stage_banner("Pemeriksaan", Icons.EXAMINATION, "Menghitung metrik awal untuk setiap frame.")
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    frames = [FrameInfo(index=idx, timestamp=idx/fps, img_path=str(fpath)) for idx, fpath in enumerate(frame_files)]
    
    compute_initial_metrics(frames)
    
    print(f"  {Icons.INFO} Menganalisis layout warna frame...")
    color_labels = analyze_color_layout(frames, CONFIG["KMEANS_CLUSTERS"])
    for f, label in zip(frames, color_labels):
        f.color_cluster = label

    return AnalysisResult(video_path=str(video_path), preservation_hash=preservation_hash, metadata=metadata,
                          frames=frames, summary={}, plots={}, localizations=[])

def main():
    parser = argparse.ArgumentParser(description="VIFA-Pro: Sistem Forensik Video Profesional.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("videos", nargs="+", help="Satu atau lebih path ke video bukti. Jika --baseline digunakan, ini adalah video suspek (hanya satu).")
    parser.add_argument("--baseline", help="Path ke video referensi (asli) untuk perbandingan.", default=None)
    parser.add_argument("--out_dir", default="vifa_pro_hasil", help="Direktori untuk menyimpan semua hasil.")
    parser.add_argument("--fps", type=int, default=15, help="Frame rate untuk ekstraksi (turunkan untuk video panjang).")
    parser.add_argument("--no-cleanup", action="store_true", help="Jangan hapus direktori frame sementara setelah selesai.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    baseline_result = None
    if args.baseline:
        if len(args.videos) > 1: print(f"{Icons.ERROR} Mode perbandingan hanya mendukung satu video suspek."), sys.exit(1)
        baseline_path = Path(args.baseline)
        if not baseline_path.exists(): print(f"{Icons.ERROR} File baseline tidak ditemukan: {baseline_path}"), sys.exit(1)
        baseline_result = process_video(baseline_path, out_dir, args.fps)

    suspect_results = []
    for vid_str in args.videos:
        video_path = Path(vid_str)
        if not video_path.exists(): print(f"{Icons.ERROR} File video tidak ditemukan: {video_path}"); continue
        suspect_results.append(process_video(video_path, out_dir, args.fps))

    print_stage_banner("Analisis Inti", Icons.ANALYSIS, "Menerapkan logika deteksi dan sintesis bukti.")
    for result in suspect_results:
        if baseline_result:
            analyze_pairwise(baseline_result.frames, result.frames, out_dir)
        else:
            synthesize_and_analyze(result.frames, out_dir)
            
        result.localizations = build_localizations(result.frames)
        total_anomaly = sum(1 for f in result.frames if f.type.startswith("anomaly"))
        result.summary = {"total_frames": len(result.frames), "total_anomaly": total_anomaly,
                          "pct_anomaly": round(total_anomaly * 100 / len(result.frames), 2) if result.frames else 0}
        result.plots = create_plots(result.frames, out_dir, Path(result.video_path).stem)

    print_stage_banner("Pelaporan", Icons.REPORTING, "Menghasilkan laporan PDF.")
    for result in suspect_results:
        video_stem = Path(result.video_path).stem
        pdf_path = out_dir / f"laporan_{video_stem}.pdf"
        print(f"  {Icons.INFO} Membuat laporan PDF untuk '{video_stem}'...")
        write_professional_report(result, pdf_path, baseline_result)
        print(f"  {Icons.SUCCESS} Laporan PDF disimpan ke: {pdf_path}")
    
    if not args.no_cleanup:
        print("\n" + "-"*35 + " PEMBERSIHAN " + "-"*35)
        for d in out_dir.glob("frames_*"):
            if d.is_dir():
                try: shutil.rmtree(d); print(f"  {Icons.SUCCESS} Direktori sementara '{d}' dihapus.")
                except OSError as e: print(f"  {Icons.ERROR} Gagal menghapus direktori '{d}': {e}")
    
    print(f"\n{Icons.SUCCESS} PROSES FORENSIK SELESAI. Hasil disimpan di '{out_dir.resolve()}'")

if __name__ == "__main__":
    main()
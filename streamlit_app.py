import streamlit as st
from pathlib import Path
import tempfile
import ForensikVideo as fv

st.set_page_config(page_title="VIFA-Pro Video Forensics")

STAGE_NAMES = {
    0: "Belum Dimulai",
    1: "Tahap 1: Pra-pemrosesan & Ekstraksi Fitur Dasar",
    2: "Tahap 2: Analisis Anomali Temporal & Komparatif",
    3: "Tahap 3: Sintesis Bukti & Investigasi Mendalam",
    4: "Tahap 4: Visualisasi & Penilaian Integritas",
    5: "Tahap 5: Penyusunan Laporan & Validasi Forensik",
    6: "Hasil Akhir Analisis Forensik"
}

# Initialize session state variables
if 'current_stage' not in st.session_state:
    st.session_state.current_stage = 0
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'video_paths_stored' not in st.session_state:
    st.session_state.video_paths_stored = False
if 'sus_video_filename' not in st.session_state:
    st.session_state.sus_video_filename = None
if 'base_video_filename' not in st.session_state:
    st.session_state.base_video_filename = None
if 'sus_video_bytes' not in st.session_state:
    st.session_state.sus_video_bytes = None
if 'base_video_bytes' not in st.session_state:
    st.session_state.base_video_bytes = None
if 'baseline_result_tahap1' not in st.session_state:
    st.session_state.baseline_result_tahap1 = None
if 'plot_bytes_for_pdf' not in st.session_state:
    st.session_state.plot_bytes_for_pdf = {}

st.title("VIFA-Pro: Investigasi Forensik Video Bertahap")

current_stage_number = st.session_state.get('current_stage', 0)
st.subheader(f"Status Proses: {STAGE_NAMES.get(current_stage_number, 'Status Tidak Diketahui')}")
st.markdown("---") # Adds a horizontal line for separation

uploaded_video = st.file_uploader("Video Bukti", type=["mp4","avi","mov","mkv"])
baseline_video = st.file_uploader("Video Baseline (Opsional)", type=["mp4","avi","mov","mkv"])
fps = st.number_input("Frame Extraction FPS", min_value=1, value=15, step=1)
run = st.button("Jalankan Analisis")

if run:
    if uploaded_video is None:
        st.error("Mohon unggah video bukti terlebih dahulu.")
    else:
        st.session_state.sus_video_filename = uploaded_video.name
        st.session_state.sus_video_bytes = uploaded_video.getbuffer()
        if baseline_video is not None:
            st.session_state.base_video_filename = baseline_video.name
            st.session_state.base_video_bytes = baseline_video.getbuffer()
        else:
            st.session_state.base_video_filename = None
            st.session_state.base_video_bytes = None
        st.session_state.video_paths_stored = True
        st.session_state.current_stage = 1
        st.experimental_rerun() # Rerun to reflect stage change immediately

# Staged display logic
if st.session_state.get('current_stage', 0) == 1:
    st.header("Tahap 1: Pra-pemrosesan & Ekstraksi Fitur Dasar")
    if not st.session_state.get('video_paths_stored', False):
        st.warning("Silakan unggah video bukti dan jalankan analisis terlebih dahulu.")
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Write evidence video from session state
            sus_path = tmpdir_path / st.session_state.sus_video_filename
            with open(sus_path, "wb") as f:
                f.write(st.session_state.sus_video_bytes)

            st.write(f"Memproses video bukti: {st.session_state.sus_video_filename}")

            # Process baseline video first if it exists (for Tahap 1 of baseline)
            if st.session_state.base_video_filename:
                base_path = tmpdir_path / st.session_state.base_video_filename
                with open(base_path, "wb") as f:
                    f.write(st.session_state.base_video_bytes)
                st.write(f"Memproses video baseline untuk Tahap 1: {st.session_state.base_video_filename}")
                baseline_result_tahap1 = fv.run_tahap_1_pra_pemrosesan(base_path, tmpdir_path, int(fps))
                st.session_state.baseline_result_tahap1 = baseline_result_tahap1
                if baseline_result_tahap1:
                    st.write("Tahap 1 untuk video baseline selesai.")
                else:
                    st.error("Gagal memproses Tahap 1 untuk video baseline.")

            # Process evidence video for Tahap 1
            result_tahap1 = fv.run_tahap_1_pra_pemrosesan(sus_path, tmpdir_path, int(fps))
            st.session_state.analysis_results = result_tahap1 # Store result for current evidence video

            if result_tahap1:
                st.success(f"Tahap 1 selesai untuk {st.session_state.sus_video_filename}.")
                st.write("Metadata:", result_tahap1.metadata)
                st.write(f"Frame diekstrak: {len(result_tahap1.frames)}")
                if st.button("Lanjutkan ke Tahap 2"):
                    st.session_state.current_stage = 2
                    st.experimental_rerun() # Trigger immediate rerun to show next stage
            else:
                st.error(f"Gagal menyelesaikan Tahap 1 untuk {st.session_state.sus_video_filename}.")

elif st.session_state.get('current_stage', 0) == 2:
    st.header("Tahap 2: Analisis Anomali Temporal & Komparatif")

    analysis_results = st.session_state.analysis_results
    baseline_result_tahap1 = st.session_state.get('baseline_result_tahap1')

    if analysis_results is None:
        st.warning("Hasil Tahap 1 tidak ditemukan. Harap jalankan Tahap 1 terlebih dahulu.")
    else:
        # Execute Stage 2 logic
        fv.run_tahap_2_analisis_temporal(analysis_results, baseline_result_tahap1)
        st.session_state.analysis_results = analysis_results # Update session state with modified results

        st.success("Tahap 2: Analisis Anomali Temporal & Komparatif selesai.")

        frames_with_ssim = sum(1 for f in analysis_results.frames if f.ssim_to_prev is not None)
        st.write(f"Jumlah frame dengan SSIM dihitung: {frames_with_ssim}")

        if baseline_result_tahap1:
            st.write("Analisis komparatif dengan baseline telah dilakukan.")
        else:
            st.write("Tidak ada video baseline untuk analisis komparatif.")

        if st.button("Lanjutkan ke Tahap 3"):
            st.session_state.current_stage = 3
            st.experimental_rerun()

elif st.session_state.get('current_stage', 0) == 3:
    st.header("Tahap 3: Sintesis Bukti & Investigasi Mendalam")
    analysis_results = st.session_state.analysis_results

    if analysis_results is None:
        st.warning("Hasil Tahap sebelumnya tidak ditemukan. Harap jalankan Tahap 1 & 2 terlebih dahulu.")
    else:
        with tempfile.TemporaryDirectory() as tmpdir_stage3:
            tmpdir_path_stage3 = Path(tmpdir_stage3)

            # Ensure the main result object is available
            if st.session_state.analysis_results: # Double check, though covered by outer if
                fv.run_tahap_3_sintesis_bukti(st.session_state.analysis_results, tmpdir_path_stage3)
                # Assuming run_tahap_3_sintesis_bukti modifies analysis_results in place or returns it
                # If it returns a new/modified object, it should be reassigned to st.session_state.analysis_results
                st.session_state.analysis_results = st.session_state.analysis_results # Explicitly mark as potentially updated

                st.success("Tahap 3: Sintesis Bukti & Investigasi Mendalam selesai.")

                # Display some info from Stage 3
                anomalies_found = sum(1 for f in st.session_state.analysis_results.frames if f.type.startswith("anomaly"))
                st.write(f"Jumlah frame yang ditandai sebagai anomali setelah Tahap 3: {anomalies_found}")

                frames_with_ela = sum(1 for f in st.session_state.analysis_results.frames if hasattr(f, 'evidence_obj') and f.evidence_obj and hasattr(f.evidence_obj, 'ela_path') and f.evidence_obj.ela_path)
                st.write(f"Jumlah frame dengan analisis ELA dilakukan: {frames_with_ela}")

                frames_with_sift = sum(1 for f in st.session_state.analysis_results.frames if hasattr(f, 'evidence_obj') and f.evidence_obj and hasattr(f.evidence_obj, 'sift_path') and f.evidence_obj.sift_path)
                st.write(f"Jumlah frame dengan analisis SIFT (untuk duplikasi) dilakukan: {frames_with_sift}")
            else:
                st.error("Objek hasil analisis tidak ditemukan di session state.")

        if st.button("Lanjutkan ke Tahap 4"):
            st.session_state.current_stage = 4
            st.experimental_rerun()

elif st.session_state.get('current_stage', 0) == 4:
    st.header("Tahap 4: Visualisasi & Penilaian Integritas")
    analysis_results = st.session_state.analysis_results

    if analysis_results is None:
        st.warning("Hasil Tahap sebelumnya tidak ditemukan. Harap jalankan Tahap 1, 2 & 3 terlebih dahulu.")
    else:
        with tempfile.TemporaryDirectory() as tmpdir_stage4:
            tmpdir_path_stage4 = Path(tmpdir_stage4)

            if st.session_state.analysis_results: # Double check
                fv.run_tahap_4_visualisasi_dan_penilaian(st.session_state.analysis_results, tmpdir_path_stage4)
                # Reassign to notify Streamlit of potential in-place modifications
                st.session_state.analysis_results = st.session_state.analysis_results

                st.session_state.plot_bytes_for_pdf = {} # Clear any old data
                plots_from_stage4 = st.session_state.analysis_results.plots
                if plots_from_stage4:
                    if 'temporal' in plots_from_stage4 and Path(plots_from_stage4['temporal']).exists():
                        with open(plots_from_stage4['temporal'], 'rb') as f_plot:
                            st.session_state.plot_bytes_for_pdf['temporal'] = f_plot.read()
                    if 'statistic' in plots_from_stage4 and Path(plots_from_stage4['statistic']).exists():
                        with open(plots_from_stage4['statistic'], 'rb') as f_plot:
                            st.session_state.plot_bytes_for_pdf['statistic'] = f_plot.read()

                st.success("Tahap 4: Visualisasi & Penilaian Integritas selesai.")

                # Display summary information
                summary = st.session_state.analysis_results.summary
                if summary:
                    st.write("Ringkasan Analisis:")
                    st.write(f"  Total Bingkai: {summary.get('total_frames', 'N/A')}")
                    st.write(f"  Total Anomali Terdeteksi: {summary.get('total_anomaly', 'N/A')}")
                    st.write(f"  Persentase Anomali: {summary.get('pct_anomaly', 'N/A')}%")

                # Display plots
                plots = st.session_state.analysis_results.plots
                if plots:
                    if 'temporal' in plots and Path(plots['temporal']).exists():
                        st.image(str(plots['temporal']), caption="Peta Anomali Temporal") # Ensure str conversion for Path
                    else:
                        st.warning("Plot temporal tidak ditemukan.")
                    if 'statistic' in plots and Path(plots['statistic']).exists():
                        st.image(str(plots['statistic']), caption="Plot Statistik") # Ensure str conversion for Path
                    else:
                        st.warning("Plot statistik tidak ditemukan.")
                else:
                    st.warning("Tidak ada plot yang dihasilkan.")
            else:
                st.error("Objek hasil analisis tidak ditemukan di session state.")

        if st.button("Lanjutkan ke Tahap 5"):
            st.session_state.current_stage = 5
            st.experimental_rerun()

elif st.session_state.get('current_stage', 0) == 5:
    st.header("Tahap 5: Penyusunan Laporan & Validasi Forensik")
    analysis_results = st.session_state.analysis_results
    baseline_result_tahap1 = st.session_state.get('baseline_result_tahap1')

    if analysis_results is None:
        st.warning("Hasil Tahap sebelumnya tidak ditemukan. Harap jalankan semua tahap sebelumnya terlebih dahulu.")
    else:
        with tempfile.TemporaryDirectory() as tmpdir_stage5:
            tmpdir_path_stage5 = Path(tmpdir_stage5)

            if st.session_state.analysis_results: # Double check
                current_plot_paths_for_pdf = {}
                if st.session_state.plot_bytes_for_pdf.get('temporal'):
                    tmp_temporal_plot_path = tmpdir_path_stage5 / "plot_temporal_for_pdf.png"
                    with open(tmp_temporal_plot_path, 'wb') as f:
                        f.write(st.session_state.plot_bytes_for_pdf['temporal'])
                    current_plot_paths_for_pdf['temporal'] = str(tmp_temporal_plot_path)

                if st.session_state.plot_bytes_for_pdf.get('statistic'):
                    tmp_statistic_plot_path = tmpdir_path_stage5 / "plot_statistic_for_pdf.png"
                    with open(tmp_statistic_plot_path, 'wb') as f:
                        f.write(st.session_state.plot_bytes_for_pdf['statistic'])
                    current_plot_paths_for_pdf['statistic'] = str(tmp_statistic_plot_path)

                # Update st.session_state.analysis_results.plots with new paths for the report
                if not hasattr(st.session_state.analysis_results, 'plots') or st.session_state.analysis_results.plots is None:
                    st.session_state.analysis_results.plots = {}

                if 'temporal' in current_plot_paths_for_pdf:
                    st.session_state.analysis_results.plots['temporal'] = current_plot_paths_for_pdf['temporal']
                if 'statistic' in current_plot_paths_for_pdf:
                    st.session_state.analysis_results.plots['statistic'] = current_plot_paths_for_pdf['statistic']

                # Pass baseline_result_tahap1 which should contain the result from baseline's Tahap 1 & 2 processing
                fv.run_tahap_5_pelaporan_dan_validasi(st.session_state.analysis_results,
                                                      tmpdir_path_stage5,
                                                      baseline_result_tahap1)
                # Reassign to notify Streamlit of potential in-place modifications
                st.session_state.analysis_results = st.session_state.analysis_results

                st.success("Tahap 5: Penyusunan Laporan & Validasi Forensik selesai.")

                pdf_path_obj = st.session_state.analysis_results.pdf_report_path
                if pdf_path_obj and Path(pdf_path_obj).exists():
                    st.write(f"Laporan PDF telah dibuat: {Path(pdf_path_obj).name}")
                    with open(pdf_path_obj, "rb") as f:
                        st.download_button(
                            label="Unduh Laporan PDF",
                            data=f,
                            file_name=Path(pdf_path_obj).name,
                            mime="application/pdf"
                        )
                else:
                    st.error("Gagal membuat atau menemukan laporan PDF.")
            else:
                st.error("Objek hasil analisis tidak ditemukan di session state.")

        if st.button("Tampilkan Hasil Akhir"):
            st.session_state.current_stage = 6
            st.experimental_rerun()

elif st.session_state.get('current_stage', 0) == 6: # Assuming stage 6 is for final results
    st.header("Hasil Akhir Analisis Forensik")
    analysis_results = st.session_state.analysis_results

    if analysis_results is None:
        st.warning("Hasil analisis tidak ditemukan. Harap jalankan semua tahap analisis terlebih dahulu.")
    else:
        st.write("Hash SHA-256 Video Bukti:", analysis_results.preservation_hash)

        summary = analysis_results.summary
        if summary:
            st.write("Ringkasan Deteksi Anomali:")
            st.write(f"  Total Bingkai Dianalisis: {summary.get('total_frames', 'N/A')}")
            st.write(f"  Total Anomali Terdeteksi: {summary.get('total_anomaly', 'N/A')}")
            st.write(f"  Persentase Anomali: {summary.get('pct_anomaly', 'N/A')}%")
        else:
            st.warning("Ringkasan analisis tidak tersedia.")

        st.info("Laporan PDF lengkap dengan detail analisis telah dibuat pada Tahap 5. Jika belum diunduh, Anda dapat kembali ke Tahap 5 untuk mengunduhnya.")
        st.info("Visualisasi grafis dari anomali (plot temporal dan statistik) telah ditampilkan pada Tahap 4.")

        if st.button("Mulai Analisis Baru"):
            # Reset relevant session state variables
            st.session_state.current_stage = 0
            st.session_state.analysis_results = None
            st.session_state.video_paths_stored = False
            st.session_state.sus_video_filename = None
            st.session_state.base_video_filename = None
            st.session_state.sus_video_bytes = None
            st.session_state.base_video_bytes = None
            st.session_state.baseline_result_tahap1 = None
            st.experimental_rerun()

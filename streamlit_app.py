import streamlit as st
from pathlib import Path
import tempfile
import ForensikVideo as fv

st.set_page_config(page_title="VIFA-Pro Video Forensics")
st.title("VIFA-Pro: Sistem Forensik Video")

uploaded_video = st.file_uploader("Video Bukti", type=["mp4","avi","mov","mkv"])
baseline_video = st.file_uploader("Video Baseline (Opsional)", type=["mp4","avi","mov","mkv"])
fps = st.number_input("Frame Extraction FPS", min_value=1, value=15, step=1)
run = st.button("Jalankan Analisis")

if run:
    if uploaded_video is None:
        st.error("Mohon unggah video bukti terlebih dahulu.")
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            sus_path = tmpdir / uploaded_video.name
            with open(sus_path, "wb") as f:
                f.write(uploaded_video.getbuffer())

            baseline_result = None
            if baseline_video is not None:
                base_path = tmpdir / baseline_video.name
                with open(base_path, "wb") as f:
                    f.write(baseline_video.getbuffer())
                baseline_result = fv.process_video(base_path, tmpdir, int(fps))

            result = fv.process_video(sus_path, tmpdir, int(fps))
            if baseline_result:
                fv.analyze_pairwise(baseline_result.frames, result.frames, tmpdir)
            else:
                fv.synthesize_and_analyze(result.frames, tmpdir)

            result.localizations = fv.build_localizations(result.frames)
            total_anom = sum(1 for f in result.frames if f.type.startswith("anomaly"))
            result.summary = {
                "total_frames": len(result.frames),
                "total_anomaly": total_anom,
                "pct_anomaly": round(total_anom * 100 / len(result.frames), 2) if result.frames else 0,
            }
            result.plots = fv.create_plots(result.frames, tmpdir, sus_path.stem)

            pdf_path = tmpdir / f"laporan_{sus_path.stem}.pdf"
            fv.write_professional_report(result, pdf_path, baseline_result)

            st.success("Analisis selesai.")
            st.write("Hash SHA-256:", result.preservation_hash)
            st.write("Total Bingkai:", result.summary["total_frames"])
            st.write(
                "Total Anomali:",
                result.summary["total_anomaly"],
                f"({result.summary['pct_anomaly']}%)",
            )
            st.image(str(result.plots["temporal"]), caption="Timeline Anomali")
            with open(pdf_path, "rb") as f:
                st.download_button("Unduh Laporan PDF", data=f, file_name=pdf_path.name)

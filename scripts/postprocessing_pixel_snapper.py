import os
import time
import gradio as gr
from modules import images, scripts_postprocessing
from modules.shared import opts
from sd_webui_pixel_snapper.snapper import process_pil_image, PixelSnapperError


def _extras_output_dir():
    return (
        getattr(opts, "outdir_extras_samples", None)
        or getattr(opts, "outpath_extras_samples", None)
        or os.path.join(getattr(opts, "data_path", ""), "output", "extras")
    )


class ScriptPostprocessingPixelSnapper(scripts_postprocessing.ScriptPostprocessing):
    name = "Pixel Snapper"
    order = 20000

    def ui(self):
        with gr.Accordion(label="Pixel Snapper", open=False):
            enabled = gr.Checkbox(label="Enable", value=False)
            k_colors = gr.Slider(
                label="Palette size (k-colors) (2–64)",
                minimum=2,
                maximum=64,
                step=1,
                value=16,
                info="Fewer colors can merge similar tones and cause details or lines to disappear. Use 16 or more to better preserve detail.",
            )
            max_dimension = gr.Slider(
                label="Max dimension",
                minimum=0,
                maximum=2048,
                step=64,
                value=0,
                info="Resize the image before processing so the longest side is at most this many pixels. Use 0 for no resize (full resolution). Lower values are faster but produce a smaller output.",
            )
        return {"enabled": enabled, "k_colors": k_colors, "max_dimension": max_dimension}

    def process(self, pp, enabled, k_colors, max_dimension):
        if not enabled:
            return
        k_colors = max(2, int(k_colors))
        max_dimension = int(max_dimension)
        original = pp.image

        try:
            outdir = _extras_output_dir()
            if outdir:
                os.makedirs(outdir, exist_ok=True)
                images.save_image(
                    original, outdir, f"pixel_snapper_original_{int(time.time() * 1000)}",
                    0, "", getattr(opts, "samples_format", "png"),
                )
        except Exception as e:
            print(f"[Pixel Snapper] Could not save original: {e}")

        try:
            pp.image = process_pil_image(original, k_colors=k_colors, max_dimension=max_dimension)
        except PixelSnapperError as e:
            print(f"[Pixel Snapper] {e}")

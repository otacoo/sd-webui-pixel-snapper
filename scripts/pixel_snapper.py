import gradio as gr
import modules.scripts as scripts
from modules import images
from modules.shared import opts
from sd_webui_pixel_snapper.snapper import process_pil_image, PixelSnapperError


class Script(scripts.Script):
    def title(self):
        return "Pixel Snapper"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("Pixel Snapper", open=False):
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
        return [enabled, k_colors, max_dimension]

    def before_process(self, p, enabled, k_colors, max_dimension):
        self._pixel_snapper_index = 0

    def postprocess_image(self, p, pp, enabled, k_colors, max_dimension):
        if not enabled:
            return
        k_colors = max(2, int(k_colors))
        max_dimension = int(max_dimension)

        idx = self._pixel_snapper_index
        self._pixel_snapper_index = idx + 1

        original = pp.image
        seed = p.all_seeds[idx] if idx < len(p.all_seeds) else (p.seed + idx)
        prompt = p.all_prompts[idx] if idx < len(p.all_prompts) else p.prompt

        try:
            images.save_image(original, p.outpath_samples, "original", seed, prompt, opts.samples_format, p=p)
        except Exception as e:
            print(f"[Pixel Snapper] Could not save original: {e}")

        try:
            pp.image = process_pil_image(original, k_colors=k_colors, max_dimension=max_dimension)
        except PixelSnapperError as e:
            print(f"[Pixel Snapper] {e}")

        try:
            images.save_image(pp.image, p.outpath_samples, "pixel_snapper", seed, prompt, opts.samples_format, p=p)
        except Exception as e:
            print(f"[Pixel Snapper] Could not save snapped: {e}")

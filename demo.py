import gradio as gr
from retrieval import retrieval
from blip import create_caption

def main(image):
    text = create_caption(image)
    results = retrieval(image, text)
    return results


if __name__ == '__main__':
    demo = gr.Interface(
        fn=main,
        inputs=gr.Image(label='Input', type='pil'),
        outputs=[
            gr.Model3D(label='Output1'),
            gr.Model3D(label='Output2'),
            gr.Model3D(label='Output3'),
            gr.Model3D(label='Output4'),
            gr.Model3D(label='Output5'),
            ]
    )
    demo.launch(share=True)

# ==============================================
# 文件：deploy_gradio.py
# ==============================================
import gradio as gr
from yolo import YOLO

# ==============================================
# 加载模型（指定 VOC 配置）
# ==============================================
yolo = YOLO(
    model_path='logs/ep100-loss2.328-val_loss2.660.pth',  # ← 改成你的 CBAM 权重路径
    input_shape=[640, 640],
    num_classes=20,  # ← VOC 是 20 类，不是 80 类！
    phi='s',
    confidence=0.5,
    nms_iou=0.45
)

# ==============================================
# 检测函数
# ==============================================
def detect(image):
    r_image = yolo.detect_image(image)
    return r_image

# ==============================================
# 创建 Gradio 界面
# ==============================================
demo = gr.Interface(
    fn=detect,
    inputs=gr.Image(type="pil", label="输入图片"),
    outputs=gr.Image(type="pil", label="检测结果"),
    title="YOLOv8 目标检测 Demo (VOC 20 类)",
    description="上传图片，自动检测目标"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0",  # ← 关键！允许外部访问
        server_port=7860,       # ← 指定端口
        share=False)             # ← 不用公网隧道

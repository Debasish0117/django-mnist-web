## Django + MNIST 在线识别（Web 项目）

本项目参考你提供的 `chapter04-01` 课程资料结构（`djangoproject/`、`vision/`、`manage.py` 等），复刻为一个可直接运行的 Django Web 项目：

- 前端上传手写数字图片
- 后端加载 CNN 模型权重进行推理
- 提供网页与 REST API 两种使用方式

> 模型权重默认复用“Django_CNN图像识别项目”训练得到的 `checkpoints/cnn_mnist.pth`。你也可以在本项目中指定任意权重路径。

---

### 目录结构

```
Django_MNIST_Web项目/
  ├─ README.md
  ├─ requirements.txt
  ├─ .env.example
  ├─ manage.py
  ├─ djangoproject/
  │   ├─ __init__.py
  │   ├─ settings.py
  │   ├─ urls.py
  │   ├─ asgi.py
  │   └─ wsgi.py
  ├─ vision/
  │   ├─ __init__.py
  │   ├─ apps.py
  │   ├─ urls.py
  │   ├─ views.py
  │   ├─ forms.py
  │   └─ templates/vision/
  │       ├─ upload.html
  │       └─ result.html
  └─ ml/
      ├─ __init__.py
      ├─ cnn.py
      └─ infer.py
```

---

### 环境准备

1) 创建并激活虚拟环境（任选其一）
- Conda:
```
conda create -n mnist_web python=3.10
conda activate mnist_web
```
- venv:
```
python -m venv .venv
.\.venv\Scripts\activate
```

2) 安装依赖
```
pip install -r requirements.txt
```

3) 配置权重路径
```
copy .env.example .env
# 编辑 .env，设置 CKPT_PATH 指向你的 CNN 权重，例如：
# CKPT_PATH=../Django_CNN图像识别项目/checkpoints/cnn_mnist.pth
```

---

### 启动

```
python manage.py migrate
python manage.py runserver 0.0.0.0:8000
```

- 网页上传与预测：访问 `http://127.0.0.1:8000/`
- REST API 预测：
```
POST http://127.0.0.1:8000/api/predict
Content-Type: multipart/form-data; name="image"
```
响应示例：`{"digit": 7}`

---

### 说明与建议

- 模型：默认使用简单 CNN 结构（与“Django_CNN图像识别项目”的 `SimpleCNN` 一致）。
- 预处理：图片先转灰度并缩放到 28x28，归一化到 MNIST 统计分布。
- 性能：CPU 上也可运行，但建议在 GPU 环境预先训练、仅在服务端加载推理。
- 安全：生产部署时关闭 `DEBUG`，限定 `ALLOWED_HOSTS`，并考虑文件上传安全与尺寸限制。



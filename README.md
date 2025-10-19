# MNIST FastAPI + React App

## Quick Start
1) Create and activate a Python venv, then install deps:
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install torch torchvision fastapi "uvicorn[standard]" pillow
```

2) Train model to produce `model/mnist_cnn.pt`:
```bash
cd model
python train.py
cd ..
```

3) Copy the model into `api/` for local run:
```bash
cp model/mnist_cnn.py api/
cp model/mnist_cnn.pt api/
# Windows:
# copy model\mnist_cnn.py api\
# copy model\mnist_cnn.pt api\
```

4) Start API:
```bash
cd api
uvicorn main:app --reload
```

5) Start Web:
```bash
cd ../web
npm install
echo "VITE_API_URL=http://127.0.0.1:8000" > .env
npm run dev
```

6) Deploy the API on Render. The Dockerfile expects `mnist_cnn.pt` to exist—train first.
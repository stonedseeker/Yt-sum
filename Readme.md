# Create a fresh environment
conda create -n test-yt python=3.11 -y
conda activate test-yt

# Install PyTorch CPU first
pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cpu

# Install requirements
cd yt-sum
pip install -r requirements.txt

# Quick test
python main.py "https://www.youtube.com/watch?v=8jPQjjsBbIc"

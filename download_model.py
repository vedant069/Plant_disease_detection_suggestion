import os
import gdown

# Create a directory called 'models'
model_dir = 'models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Model 1 - Download from Google Drive
model_1_url = 'https://drive.google.com/uc?id=1R561Xcfy85SqB5AQQm3GNSTvQecpbUBe'
model_1_path = os.path.join(model_dir, 'export.pkl')  # Replace with original filename and extension
gdown.download(model_1_url, model_1_path, quiet=False)

# Model 2 - Download from Google Drive
model_2_url = 'https://drive.google.com/uc?id=16PO19ah0Xuu05dd0nnQLG2VyXwmgtV0G'
model_2_path = os.path.join(model_dir, 'luna-ai-llama2-uncensored.ggmlv3.q4_K_S.bin')  # Replace with original filename and extension
gdown.download(model_2_url, model_2_path, quiet=False)

print(f"Models downloaded to {model_dir} folder")

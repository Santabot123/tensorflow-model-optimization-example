import zipfile
from zipfile import ZipFile
import glob
import os.path



models_names=glob.glob('./models/*')
for name in models_names:
    basename=os.path.splitext(os.path.basename(name))[0]
    with ZipFile(f'./ziped_models/{basename}.zip', 'w',compression=zipfile.ZIP_DEFLATED) as zip_object:
       # Adding files that need to be zipped
       zip_object.write(f'./models/{basename}.tflite')


import zipfile
import os

path = r"./ReutersCorpusVolume1/Data/ReutersCorpusVolume1_Original/CD1/"
list = os.listdir(path)

target_path = r"./xml"
isExists = os.path.exists(target_path)
if not isExists:
    os.makedirs(target_path)

for z in list:
    file_path = os.path.join(path,z)
    zipf = zipfile.ZipFile(file_path)
    zipf.extractall(target_path)
    zipf.close()

path = r"./ReutersCorpusVolume1/Data/ReutersCorpusVolume1_Original/CD2/"
list = os.listdir(path)

for z in list:
    file_path = os.path.join(path,z)
    zipf = zipfile.ZipFile(file_path)
    zipf.extractall(target_path)
    zipf.close()

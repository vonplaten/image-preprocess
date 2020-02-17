import pandas as pd
import torchvision
import os, shutil
import numpy as np

import conf
import ctransform
import csample

class Datacreator():
    def __init__(self, rmdir, train, val):
        self._config = conf.DataConfig()
        self.dftrain = None
        self.dfval = None
        self.dotrain = train
        self.doval = val
        for p in ["train", "valid"]:
            if rmdir and ((p=="train" and self.dotrain) or (p=="valid" and self.doval)):
                traincrops = os.path.join(self._config.datapath, f"{p}crops")
                if os.path.exists(traincrops):
                    shutil.rmtree(traincrops)
                os.makedirs(traincrops)
                if os.path.isfile(os.path.join(traincrops, "crops.csv")):
                    os.remove(os.path.join(traincrops, "crops.csv"))
                open(os.path.join(traincrops, "crops.csv"), "w").close()

        if (self._config.allfilepath != None):
            # ImageId_ClassId,EncodedPixels
            df_all = pd.read_csv(self._config.allfilepath)  # (50272, 2), ['ImageId_ClassId', 'EncodedPixels']
            df_all['ImageId'], df_all['ClassId'] = zip(
                *df_all['ImageId_ClassId'].str.split('_'))  # ['ImageId_ClassId', 'EncodedPixels', 'ImageId', 'ClassId']
            df_all = df_all.drop(["ImageId_ClassId"], axis=1)
            # https://www.shanelynn.ie/summarising-aggregation-and-grouping-data-in-python-pandas/
            # https://stackoverflow.com/questions/30679467/pivot-tables-or-group-by-for-pandas
            df_all['ClassId'] = df_all['ClassId'].astype(int)
            df_all = df_all.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')  # [1, 2, 3, 4]
            # slow! df_all = df_all.groupby(["ImageId", "ClassId"]).agg(classnr=("EncodedPixels", lambda x: x if x.count()>0 else None )).unstack("ClassId")
            df_all.columns = [f"class_{col}" for col in df_all.columns]
            df_all['defects'] = df_all.count(axis=1)  # [1, 2, 3, 4, 'defects']
            # df_all["imname"] = df_all.index
            df_all = df_all.reset_index()
            # df_all["ImagePath"] = df_all["ImageId"].apply(lambda x: f"{self._config.trainimagesfolderpath}{x}")
            # self.df_train = self.df_train.loc[ (self.df_train["defects"]== 1) ]

            # split 12000 images
            train = df_all.sample(frac=0.85, random_state=200)
            self.dfval = df_all.drop(train.index)
            #del df_all  # cleaning

            # 5 classes
            t1 = train[pd.isna(train["class_1"]) == False] #803
            t2 = train[pd.isna(train["class_2"]) == False] #223
            t3 = train[pd.isna(train["class_3"]) == False] #4634
            t4 = train[pd.isna(train["class_4"]) == False] #718
            tdrop = list(set(list(t1.index)+list(t2.index)+list(t3.index)+list(t4.index)))
            t0 = train.drop(tdrop)
            del train

            #self.dftrain = t3.loc[t3.index == 67] ###########################
            #self.dftrain = self.dftrain.sample(4, replace=True)

            # balance
            # if len(t1) > 0:
            #     t1 = t1.sample(len(t3), replace=True)
            # if len(t2) > 0:
            #     t2 = t2.sample(len(t3), replace=True)
            # # if len(t3) > 0:
            # #     t3 = t3.sample(len(t3)*4, replace=True)
            # if len(t4) > 0:
            #     t4 = t4.sample(len(t3), replace=True)
            t0 = t0.sample(frac=0.2, random_state=200)
            # print(f"t0:{len(t0)}, t1:{len(t1)}, t2:{len(t2)}, t3:{len(t3)}, t4:{len(t4)}")
            self.dftrain = pd.concat([t0, t1, t2, t3, t4], ignore_index=True)
            self.dftrain = self.dftrain.sample(frac=1, random_state=200) #shuffle
            #self.dftrain = self.dftrain.sample(20, replace=True)
            self.dftrain.reset_index(inplace=True)

            #self.dfval = self.dfval[pd.isna(self.dfval["class_3"]) == False]
            #self.dfval = self.dfval.sample(20, replace=True)
            self.dfval.reset_index(inplace=True)
            print("Loaded source data")

    def crop(self):
        if self.dotrain:
            traintransform = torchvision.transforms.Compose([
                ctransform.CropCoord100((224, 224), cover_percent=100, use_empty_masks=0.2, min_pixels=800, keep_sample=False),
                ctransform.Save(os.path.join(self._config.datapath, "traincrops"))
            ])
            traincounter = 0
            classes = np.zeros(self._config.numclasses + 1).astype(np.uint16)
            print(f"Cropping {len(self.dftrain)} train images.")
            for index, row in self.dftrain.iterrows():
                sample = csample.Sample(
                    id_="original",
                    folder=self._config.trainimagesfolderpath,
                    filename=row.loc["ImageId"],
                    originalsize=(256, 1600),
                    rle=row.iloc[self.dftrain.columns.get_loc("class_1") : self.dftrain.columns.get_loc("class_4")+1].values.tolist(),
                )
                samplelist, classcounter = traintransform(sample)
                traincounter += len(samplelist)
                classes += classcounter
                if index%500==0:
                    print(f"Train_images:{index} - Crops:{traincounter} - Classes:{classes}")
            print(f"Total - Train_images:{index} - Crops:{traincounter} - Classes:{classes}\n")

        if self.doval:
            valcounter = 0
            classes = np.zeros(self._config.numclasses + 1).astype(np.uint16)
            valtransform = torchvision.transforms.Compose([
                ctransform.CropCoord100((224, 224)),
                ctransform.Save(os.path.join(self._config.datapath, "validcrops"))
            ])
            print(f"Cropping {len(self.dfval)} validation images.")
            for index, row in self.dfval.iterrows():
                sample = csample.Sample(
                    id_="original",
                    folder=self._config.trainimagesfolderpath,
                    filename=row.loc["ImageId"],
                    originalsize=(256, 1600),
                    rle=row.iloc[self.dfval.columns.get_loc("class_1") : self.dfval.columns.get_loc("class_4")+1].values.tolist(),
                )
                samplelist, classcounter = valtransform(sample)
                valcounter += len(samplelist)
                classes += classcounter
                if index%500==0:
                    print(f"Validation_images:{index} - Crops:{valcounter} - Classes:{classes}")
            print(f"Total - Validation_images:{index} - Crops:{valcounter} - Classes:{classes}\n")

        
if __name__ == "__main__":
    datacreator = Datacreator(rmdir=False, train=False, val=False)
    datacreator.crop()
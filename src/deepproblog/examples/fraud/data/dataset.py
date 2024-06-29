import os
import torch
import torchvision.transforms as transforms

from deepproblog.dataset import ImageDataset, Dataset, ABC
from deepproblog.query import Query
from problog.logic import Term, Constant
import pandas as pd

path = os.path.dirname(os.path.abspath(__file__))

transform = transforms.Compose([transforms.ToTensor()])


class TabularDataset(Dataset, ABC):
    def __init__(self, files, columns, target, transform=None):
        super().__init__()
        df_standard = pd.read_csv(files[0])
        df_standard = df_standard.loc[:, ~df_standard.columns.str.contains("^Unnamed")]
        df_standard = df_standard.loc[:, ~df_standard.columns.str.contains("^index")]
        df_scaled = pd.read_csv(files[1])
        df_scaled = df_scaled.loc[:, ~df_scaled.columns.str.contains("^Unnamed")]
        df_scaled = df_scaled.loc[:, ~df_scaled.columns.str.contains("^index")]
        self.df_standard = df_standard
        if columns:
            self.x_data_scaled = torch.tensor(
                df_scaled[columns].astype(float).values
            ).float()
        else:
            columns_to_keep = list(df_scaled.columns)
            columns_to_keep.remove(target)
            self.x_data_scaled = torch.tensor(
                df_scaled[columns_to_keep].astype(float).values
            ).float()
        self.data = df_standard[[target]]
        # self.data_tensor = torch.tensor(self.data['targets'].values)
        self.columns = columns
        self.target = target
        self.transform = transform

    def __getitem__(self, index):
        if type(index) is tuple:
            index = index[0]
        # out = [self.x_data[column][index] for column in self.columns]
        """
        if self.transform:
            out = self.transform(out)
        """
        out = self.x_data_scaled[int(index)]
        return out


class FraudDataset(TabularDataset):
    def __init__(self, subset, columns, target, files=None):
        if not files:
            if subset == "Train":
                files = [
                    "{}/fraud{}PreprocessedDownsampled.csv".format(path, subset),
                    "{}/fraud{}PreprocessedScaledDownsampled.csv".format(path, subset),
                ]
            else:
                files = [
                    "{}/fraud{}PreprocessedOriginal.csv".format(path, subset),
                    "{}/fraud{}PreprocessedScaledOriginal.csv".format(path, subset),
                ]
        super().__init__(files, columns=columns, target=target, transform=transform)
        self.subset = subset

    def is_noisy(self, i):
        if "noisy" in self.df_standard.columns:
            return self.df_standard["noisy"][i]
        else:
            return False

    def get_rule_values(self, i):
        amt = self.df_standard["amt"][i]
        hour = self.df_standard["hour"][i]
        category_grocery_pos = self.df_standard["category_grocery_pos"][i]

        if amt >= 1262.425:
            out_amt = "ExtremeAmount"
        elif amt >= 695.445:
            out_amt = "BigAmount"
        elif amt > 259.04:
            out_amt = "MediumAmount"
        elif amt >= 0:
            out_amt = "LittleAmount"
        else:
            out_amt = "_"

        if hour <= 3.5 and hour >= 0:
            out_hour = "EarlyMorning"
        elif hour <= 21.5 and hour >= 0:
            out_hour = "Day"
        elif hour >= 0:
            out_hour = "Night"
        else:
            out_hour = "_"

        if category_grocery_pos == 1:
            out_category_grocery_pos = "CategoryGroceryPos"
        elif category_grocery_pos == 0:
            out_category_grocery_pos = "CategoryNonGroceryPos"
        else:
            out_category_grocery_pos = "_"

        return out_amt, out_hour, out_category_grocery_pos

    def to_query(self, i):
        if str(self.data[self.target][i]) == "1":
            is_fraud = "fraud"
        elif str(self.data[self.target][i]) == "0":
            is_fraud = "non_fraud"
        else:
            is_fraud = "unknown"
        amt, hour, category_grocery_pos = self.get_rule_values(i)
        noisy = "is_noisy" if self.is_noisy(i) else "is_not_noisy"
        sub = {Term("x"): Term("tensor", Term(self.subset, Constant(i)))}
        if is_fraud == "unknown":
            out = Query(
                Term(
                    "predict_fraud",
                    Term("x"),
                    Term(amt),
                    Term(hour),
                    Term(category_grocery_pos),
                ),
                sub,
            )
        else:
            out = Query(
                Term(
                    "predict_fraud",
                    Term("x"),
                    Term(amt),
                    Term(hour),
                    Term(category_grocery_pos),
                    Term(noisy),
                    Term(is_fraud),
                ),
                sub,
            )
        return out

    def __len__(self):
        return len(self.data)

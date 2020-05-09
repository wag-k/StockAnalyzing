## -*- coding: utf-8 -*-
"""
Created on Thu Apr 09

author: Kenta Kawaguchi
"""

#### 標準 or 外部ライブラリ ####
from bs4 import BeautifulSoup
from collections import OrderedDict, deque
import datetime
import glob
import json
import math
import matplotlib.pyplot as plt   
import numpy as np
import pandas as pd
import os
import random
import ssl
import sys
import time
import urllib

from bs4 import BeautifulSoup
#### 自作 ####
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../MachineLearning'))
from MachineLearning import neural_network
 
class StockAnalyzing:
    def __init__(self, dname = "./data/"):
        #### カレントディレクトリの調整 ####
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        print(os.getcwd())

        #### 定数 ####
        self.data_columns = ['date','open','high','low','close','valume','adjclose']

        #### メンバ ####
        self.dname = dname
        with open(os.path.join(dname,"stock_list.json"), encoding="utf-8") as f:
            self.stock_list = json.load(f, object_pairs_hook=OrderedDict)
        
    # CSVを読み込んでリストをpd.DataFrameに格納
    def get_stock_list_from_csv(self):
        self.stockname_list = pd.read_csv(os.path.join(self.dname, "stocklist.csv"))

    #### 手動で銘柄リストに登録するときに使用 ####
    def register_stock_list_by_manual(self):
        while(True):
            #### code ####
            while(True):
                code = input("Plz input code (q: quit): ")
                if code == "q":                                
                    with open(self.dname+"/stock_list.json", "w", encoding="utf-8") as f:
                        json.dump(self.stock_list, f, indent=4, ensure_ascii=False)
                    return True
                if (code.isdigit()):
                    code = int(code)
                    break

            #### name ####
            name = input("Plz input name (q: quit): ")
            if name == "q":
                with open(self.dname+"/stock_list.json", "w") as f:
                    json.dump(self.stock_list, f, indent=4, ensure_ascii=False)
                    json.dump(self.stock_list, f, indent=4, ensure_ascii=False)
                return True
            
            #### add ####
            print(str(code), ": ", name)
            judge = input("Enter y/Y if OK.")
            if (judge.lower() == "y"):
                self.add_stock_list(code, name)

    #### JSON用の銘柄リストに追加 ####
    def add_stock_list(self, code, name):
        self.stock_list[code] = {"Code":code, "Name": name}


    #### 自動スクレイピング #####
    def scrape_stcok_data_old_data(self):
        start_year = 1983
        end_year = 2020
        # リストの何番目からスクレイピングを開始するか指定
        while True:
            try:
                start_code = input("Please input start stock code (q: quit): ")
                if start_code == "q":
                    return True
                start_code = int(start_code)
            except Exception as e:
                print("Error: Invalid input.")
                continue
            else:
                break
        scraping_list = self.stockname_list.loc[start_code:]
        for code in scraping_list["Code"]:
            for year in range(start_year, end_year+1):
                print("Code: ", str(code), ", Year: ", str(year))
                data =  self.get_stock_data(code, year)
                if type(data) == type(False):
                    if not data:
                        print("Failed to get data.")
                        continue # データ取得でエラーが出たら次
                    else:
                        print("Error: Unexpected data type")
                        return
                dname = os.path.join(self.dname, str(code))
                if self.save_stock_data2csv(code, year, data, dname):
                    print("Success")
                else:
                    print("Failed to get save data.")
                time.sleep(1+random.randint(0, 5)) # ランダムな待ち時間で人間らしく
        return True

    # ダウンロードした株データをCSV形式で出力 ####
    def save_stock_data2csv(self, code, year, df, path):
        os.makedirs(path, exist_ok=True)
        fname = str(code)+"_"+str(year)+".csv"
        df.to_csv(os.path.join(path, fname))
        return True

    #### 株式データの取得 ####
    #### データの取得に失敗したらFalseを返す ####
    #### 参考： https://manareki.com/stock_python_scraping ####
    def get_stock_data(self, code, year):
        url = "https://kabuoji3.com/stock/"+str(code)+"/"+str(year)+"/" 
        ssl._create_default_https_context = ssl._create_unverified_context
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                html = response.read() 
        except urllib.error.URLError as e:
            print(e.reason)    
            return False
        except urllib.error.HTTPError as e:
            print(e.reason)    
            return False
        except Exception as e:
            print("Unexpected Error")
            return False
        try:
            soup = BeautifulSoup(html, "html.parser")
            time.sleep(random.randint(1,5))
            stockdata= soup.find_all("td")
            stockdata = [s.contents[0] for s in stockdata]
            stockdata = list(zip(*[iter(stockdata)]*7))
            df = pd.DataFrame(stockdata,)
            df.columns= self.data_columns
        except Exception as e:
            print("Error: Invalid data format")
            return False
        return df

    #### 条件にマッチする銘柄リストを生成 ####
    def make_code_list(self, condition = ["this_year", "not_yet"]):
        #### ディレクトリ以外をリストから消しておく ####
        code_list = [code for code in os.listdir(self.dname) if  os.path.isdir(os.path.join(self.dname, code))]

        #### スクレイピング中のデータは無視 ####
        if "this_year" in condition: 
            this_year = datetime.datetime.now().year
            code_list = [code for code in code_list if (code+"_"+str(int(this_year))+".csv") in os.listdir(os.path.join(self.dname, code))]

        #### まだcode_history.csvが生成されていない銘柄を処理 ####
        if "not_yet" in condition:
            code_list = [code for code in code_list if not "code_history.csv" in os.listdir(os.path.join(self.dname, code))]
        return code_list

    #### リスト内の銘柄フォルダ中のCSV形式データを結合、code_history.csvに出力。 ####
    def make_full_timeline_data(self, code_list):
        for code in code_list:
            fulldata = pd.DataFrame(index=[], columns= self.data_columns)
            now_path = os.path.join(self.dname, code)
            print("Make full-timeline file: ",now_path)
            for fname in os.listdir(now_path):
                fpath = os.path.join(now_path, fname)
                if os.path.isfile(fpath):
                    if ".csv" in fpath:
                        fulldata = pd.concat([fulldata, pd.read_csv(fpath, usecols=[i+1 for i in range(len(self.data_columns))])])
            fulldata.to_csv(os.path.join(now_path,"code_history.csv"))
        return True

    def struct_nn(self, net_size, weight_init, optimiser = "SGD", init_weight = "std", activation = "ReLU"):
        self.nn   = neural_network.NeuralNetwork(optimiser = optimiser, init_weight = init_weight, activation = activation)
        self.nn.struct_network(net_size, weight_init)
        self.nn.show_params_shape()

    def train_nn(self, batch_size, iters_num, learning_rate):
        # データ数が多すぎて全データを一度にNNへ入力できないため、
        # あらかじめStockAnalyzingクラスでMini_batchを作って学習させる。
        # したがってtrainの引数のmini_batchはFalse、iters_numは1
        # 学習結果は
        a_input = np.array([abs(math.sin(n/6))*np.random.random_sample(input_size[0]) for n in range(n_data)])
        ans = np.array([0, 1, 0, 0, 0]) 
        self.nn.train(a_input, ans, 1, learning_rate, train_method= "grad", mini_batch= batch_size, activation= "ReLU", err_func= "cross_entropy")

    # select_code_for_train_testによって得られた、データ数の担保が取れた銘柄のみ入力すること。
    # 入力が不正な場合はFalseを返す
    def extract_data_for_train_test(self, data_num, code):
        now_path = os.path.join(self.dname, code)
        fpath = os.path.join(now_path, "code_history.csv")
        #### 入力確認 ####
        if not os.path.isfile(fpath):
            print("Error: Invalid filepath, ", fpath)
            return False
        stock_data_df = pd.read_csv(fpath, usecols=[i+2 for i in range
        (len(self.data_columns)-1)])
        if not data_num <= len(stock_data_df):
            print("Error: Lack of data, ", len(stock_data_df))
            return False

        ##### データ抽出 ####
        margin = len(stock_data_df) - data_num
        idx_start = random.randint(0, margin)
        stock_data_df = stock_data_df[idx_start:idx_start+data_num]
        code_row = np.full((data_num, 1), code)
        stock_data = np.block([code_row, stock_data_df.values])
        return stock_data



    def select_code_for_train_test(self, data_num):
        #### ディレクトリ以外をリストから消しておく ####
        code_list_all = [code for code in os.listdir(self.dname) if  os.path.isdir(os.path.join(self.dname, code))]
        code_list = []

        #### データ数が足りている銘柄をリストに追加 ####
        for code in code_list_all:        
            now_path = os.path.join(self.dname, code)
            fpath = os.path.join(now_path, "code_history.csv")
            if os.path.isfile(fpath):
                now_data_df = pd.read_csv(fpath, usecols=[i+2 for i in range(len(self.data_columns)-1)])
                if data_num <= len(now_data_df):
                    code_list.append(code)
                else:
                    continue
            else:
                continue
        return code_list

    

def main():

    batch_size = 10
    iters_num = 100
    learning_rate = 0.1
    weight_init = 1
    input_size = [7] # code,open,high,low,close,valume,adjclose
    output_size = [201] # -50%以下, -49.5, ... -0.5, 0, 0.5, ... 49.5, 50以上  
    net_size = input_size + [4096, 1024, 256, 64] + output_size # ハイパーパラメータ
    data_num = 4981

    stock = StockAnalyzing()
    stock.get_stock_list_from_csv()
    code_list = stock.make_code_list()
    stock.struct_nn(net_size, weight_init)
    code_list_for_train = stock.select_code_for_train_test(data_num)
    print(code_list_for_train)
    for code in code_list_for_train:
        print(stock.extract_data_for_train_test(data_num, code))
    #stock.make_full_timeline_data(code_list)
    #stock.scrape_stcok_data_old_data()
    #stock.register_stock_list_by_manual()

    return True


if __name__ == "__main__":
    main()
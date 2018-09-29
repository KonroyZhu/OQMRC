import pickle

train_obj=pickle.load(open("esm_record/train.pkl","rb"))
dev_obj=pickle.load(open("esm_record/dev.pkl","rb"))
print(len(train_obj.keys()))
for key in train_obj.keys():
    print(key,train_obj[key])

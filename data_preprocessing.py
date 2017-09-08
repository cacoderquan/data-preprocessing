

import numpy as np
from scipy import sparse
import pandas as pd
import xgboost as xgb
import re
import string
import time
import seaborn as sns

from sklearn import preprocessing, pipeline, metrics, model_selection
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
get_ipython().magic(u'matplotlib inline')


from bokeh.io import output_file, show,gridplot
from bokeh.charts import Scatter,Histogram,Bar, show
from bokeh.plotting import figure

from bokeh.models import (
  GMapPlot, GMapOptions, ColumnDataSource, Circle, DataRange1d, PanTool, WheelZoomTool, BoxSelectTool
)
from bokeh.io import output_notebook
output_notebook()




train_data = pd.read_json('../input/train.json')
test_data = pd.read_json('../input/test.json')



train_data['target'] = train_data['interest_level'].apply(lambda x: 0 if x=='low' else 1 if x=='medium' else 2)
train_data['low'] = train_data['interest_level'].apply(lambda x: 1 if x=='low' else 0)
train_data['medium'] = train_data['interest_level'].apply(lambda x: 1 if x=='medium' else 0)
train_data['high'] = train_data['interest_level'].apply(lambda x: 1 if x=='high' else 0)




full_data=pd.concat([train_data,test_data])



num_vars = ['bathrooms','bedrooms','latitude','longitude','price']
cat_vars = ['building_id','manager_id','display_address','street_address']
text_vars = ['description','features']
date_var = 'created'
image_var = 'photos'
id_var = 'listing_id'


# # Exploratory Data Analysis


sns.distplot(full_data.price)



num_stats = train_data[num_vars].describe().reset_index()

price_25_qt = num_stats.query('index=="25%"')['price'].values[0]
price_75_qt = num_stats.query('index=="75%"')['price'].values[0]

lat_25_qt = num_stats.query('index=="25%"')['latitude'].values[0]
lat_75_qt = num_stats.query('index=="75%"')['latitude'].values[0]

lon_25_qt = num_stats.query('index=="25%"')['longitude'].values[0]
lon_75_qt = num_stats.query('index=="75%"')['longitude'].values[0]




sns.distplot(full_data.query('price >%f and price <%f' % (price_25_qt, price_75_qt)).price)





sns.boxplot(full_data.price, orient='v', width=0.2)




sns.boxplot(full_data.price, showfliers=False, orient='v', width=0.2)



sns.jointplot("longitude","latitude", 
              data=train_data.query('latitude>%f and latitude<%f and longitude>%f and longitude<%f' % 
                                    (lat_25_qt, lat_75_qt, lon_25_qt, lon_75_qt)
                                   )
             )



sns.jointplot("longitude","latitude", 
              data=train_data.query('latitude>%f and latitude<%f and longitude>%f and longitude<%f' % 
                                    (lat_25_qt, lat_75_qt, lon_25_qt, lon_75_qt)
                                   ),
              kind='kde'
             )


sns.lmplot("longitude","latitude", 
              data=train_data.query('latitude>%f and latitude<%f and longitude>%f and longitude<%f' % 
                                    (lat_25_qt, lat_75_qt, lon_25_qt, lon_75_qt)
                                   ),
           hue="target",
           fit_reg=True
          )




sns.pairplot(train_data,
             x_vars=num_vars + ['target'],
             y_vars=num_vars + ['target'],
             hue="target")


sns.pairplot(train_data.query('price>%f and price<%f and latitude>%f and latitude<%f and longitude>%f and longitude<%f' % 
                                    (price_25_qt, price_75_qt, lat_25_qt, lat_75_qt, lon_25_qt, lon_75_qt)
                             ),
             x_vars=num_vars + ['target'],
             y_vars=num_vars + ['target'],
             hue="target")



# In[16]:

sns.lmplot(x="bedrooms", y="price", 
          data=full_data.query('price >%f and price <%f' % (price_25_qt, price_75_qt)),
          hue="target")



sns.boxplot(x='interest_level',
               y='price',
               data=train_data.query('price>%f and price<%f' % 
                                    (price_25_qt, price_75_qt)
                                   )
              )




sns.violinplot(x='interest_level',
               y='price',
               data=train_data.query('price>%f and price<%f' % 
                                    (price_25_qt, price_75_qt)
                                   )
              )



from bokeh.io import output_file, show
from bokeh.models import (
  GMapPlot, GMapOptions, ColumnDataSource, Circle, DataRange1d, PanTool, WheelZoomTool, BoxSelectTool
)

map_options = GMapOptions(lat=40.7145,  lng=-73.9425, map_type="roadmap", zoom=11)

plot = GMapPlot(
    x_range=DataRange1d(), y_range=DataRange1d(), map_options=map_options
)
plot.title.text = "Newyork"


plot.api_key = "YOURGOOGLEAPIKEY"


source_high = ColumnDataSource(
    data=dict(
        lat=full_data.query('interest_level=="high"')["latitude"].tolist(),
        lon=full_data.query('interest_level=="high"')["longitude"].tolist()
    )
)
circle_high = Circle(x="lon", y="lat", size=3, fill_color="red", fill_alpha=0.3, line_color=None)

source_medium = ColumnDataSource(
    data=dict(
        lat=full_data.query('interest_level=="medium"')["latitude"].tolist(),
        lon=full_data.query('interest_level=="medium"')["longitude"].tolist()
    )
)
circle_medium = Circle(x="lon", y="lat", size=2, fill_color="blue", fill_alpha=0.3, line_color=None)

source_low = ColumnDataSource(
    data=dict(
        lat=full_data.query('interest_level=="low"')["latitude"].tolist(),
        lon=full_data.query('interest_level=="low"')["longitude"].tolist()
    )
)
circle_low = Circle(x="lon", y="lat", size=2, fill_color="green", fill_alpha=0.3, line_color=None)

plot.add_glyph(source_high, circle_high)
plot.add_glyph(source_medium, circle_medium)
plot.add_glyph(source_low, circle_low)

plot.add_tools(PanTool(), WheelZoomTool(), BoxSelectTool())
output_file("gmap_plot.html")
show(plot)


# # Conclusion

train_size = train_data.shape[0]

full_data['created_datetime'] = pd.to_datetime(full_data['created'], format="%Y-%m-%d %H:%M:%S")
full_data['created_epoch']=full_data['created_datetime'].apply(lambda x:x.value//10**9)
date_num_vars = ['created_epoch']

LBL = preprocessing.LabelEncoder()
LE_vars=[]
LE_map=dict()
for cat_var in cat_vars:
    print ("Label Encoding %s" % (cat_var))
    LE_var=cat_var+'_le'
    full_data[LE_var]=LBL.fit_transform(full_data[cat_var])
    LE_vars.append(LE_var)
    LE_map[cat_var]=LBL.classes_
    
print ("Label-encoded feaures: %s" % (LE_vars))




full_vars = num_vars + date_num_vars + LE_vars

train_x = full_data[full_vars][:train_size]
train_y = full_data['target'][:train_size].values

test_x = full_data[full_vars][train_size:]
test_y = full_data['target'][train_size:].values

print ("training data size: ", train_x.shape,"testing data size: ", test_x.shape)


params = dict()
params['objective'] = 'multi:softprob'
params['num_class'] = 3
params['eta'] = 0.1
params['max_depth'] = 6
params['min_child_weight'] = 1
params['subsample'] = 0.7
params['colsample_bytree'] = 0.7
params['gamma'] = 1
params['seed']=1234

cv_results = xgb.cv(params, 
                    xgb.DMatrix(train_x, label=train_y.reshape(train_x.shape[0],1)),
                    num_boost_round=1000000, 
                    nfold=5,
                    metrics={'mlogloss'},
                    seed=1234,
                    callbacks=[xgb.callback.early_stop(50)],
                    verbose_eval=50
                   )

best_score = cv_results['test-mlogloss-mean'].min()
best_iteration = len(cv_results)
print ('Best iteration: %d, best score: %f' % (best_iteration, best_score))


# ## Training


start = time.time()
clf = xgb.XGBClassifier(learning_rate = 0.1
                        , n_estimators =best_iteration
                        , max_depth = 6
                        , min_child_weight = 1
                        , subsample = 0.7
                        , colsample_bytree = 0.7
                        , gamma = 1
                        , seed = 1234
                        , nthread = -1
                       )

clf.fit(train_x, train_y)

print ("Training finished in %d seconds." % (time.time()-start))

feature_importance = pd.DataFrame(sorted(zip(full_vars,clf.feature_importances_)
                                         , key=lambda x: x[1], reverse = True)
                                  ,columns=['feature_name','importance']) 

feature_importance.sort_values('importance', 
                               ascending=True).plot.barh(x='feature_name',
                                                        figsize=(3,5))


# ## Predicting


preds = clf.predict_proba(test_x)
sub_df = pd.DataFrame(preds,columns = ["low", "medium", "high"])
sub_df["listing_id"] = test_data.listing_id.values
sub_df.to_csv("../output/my_first_submission.csv", index=False)



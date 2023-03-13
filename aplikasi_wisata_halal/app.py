from flask import *

import sqlite3
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder

from math import *
import geocoder

le = LabelEncoder()

def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan

conn = sqlite3.connect('../../tugas-akhir.db')
cursor = conn.cursor()
sql_meta = "SELECT id_tempat,jenis, coordinat, tempat,city,img, description FROM data"
cursor.execute(sql_meta)
result_meta = cursor.fetchall()
def meta_data():
    # content-based
    meta = pd.DataFrame(result_meta,columns= ['id_tempat_rating','jenis','coordinat','tempat','kota','src_img','deskripsi'])
    meta = meta.dropna(subset=['jenis'])

    xid_city = meta['kota']

    conn = sqlite3.connect('../../tugas-akhir.db')
    cursor = conn.cursor()

    sql_trend = "SELECT tempatR, COUNT(*), AVG(rating) FROM ratings GROUP BY tempatR"
    cursor.execute(sql_trend)
    result_trend = cursor.fetchall()
    trend = pd.DataFrame(result_trend,columns= ['tempat','count_tempat','avg_rating'])
    trend['tempat'] = trend['tempat'].str.title()
    # meta['id_tempat'] = le.fit_transform(np.ravel(xid_city))
    meta['id_tempat'] = le.fit_transform(meta['kota'])
    meta['id_tempat'] = meta['id_tempat'].apply(convert_int)

    meta['deskripsi'] = meta['deskripsi'].fillna(value="")

    trend['count_tempat'] = trend['count_tempat'].apply(convert_int)
    meta['tempat'] = meta['tempat'].str.title()
    # left join
    meta = pd.merge(meta,trend,on='tempat',how='left')
    meta['avg_rating'] = meta['avg_rating'].fillna(0)
    meta['avg_rating'] = meta['avg_rating'].round(1)

    
    return meta




# # # print(store.info())
# # # print(store.id_tempat.value_counts())
def koneksi():
    con = sqlite3.connect('../../tugas-akhir.db')
    cur = con.cursor()
    return cur
def haversine_distance(lat1, lon1, lat2, lon2):
   r = 6371
   phi1 = np.radians(lat1)
   phi2 = np.radians(lat2)
   delta_phi = np.radians(lat2 - lat1)
   delta_lambda = np.radians(lon2 - lon1)
   a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) *   np.sin(delta_lambda / 2)**2
   res = r * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
   return np.round(res, 2)

g = geocoder.ip('me')
curr_lat = g.lat
curr_lon = g.lng

sql_maps = "SELECT tempat,city,coo1,coo2 FROM data"
cursor.execute(sql_maps)
result_maps = cursor.fetchall()

start_lat, start_lon = curr_lat, curr_lon
    
cities = pd.DataFrame(result_maps,columns= ['tempat','City','Lat','Lon'])
# print(cities.info)
cities['tempat'] = cities['tempat'].str.title()
distances_km = []
for row in cities.itertuples(index=False):
    distances_km.append(
        haversine_distance(start_lat, start_lon, row.Lat, row.Lon)
    )
cities['Distance'] = distances_km


def check_user(id_user, password):
    con = sqlite3.connect('../../tugas-akhir.db')
    cur = con.cursor()
    cur.execute('Select id_user,password FROM user WHERE id_user=? and password=?', (id_user, password))

    result = cur.fetchone()
    if result:
        return True
    else:
        return False
# 105496414599981678036
# newacc123
app = Flask(__name__)
app.secret_key = "@adenjmn"

@app.route("/")
def home():
    arr_meta = meta_data().values.tolist()
    
    if 'id_user' in session:
        msg = "Berhasil masuk"
        user_id = session["id_user"]

        cursor = koneksi()
        cursor.execute('SELECT id_user,img_src FROM user WHERE id_user=? order by id_user limit 1',(user_id,))
        user = cursor.fetchall()
        return render_template('index.html', _meta=arr_meta,msg=msg,user=user)
    else:
        return render_template('index.html', _meta=arr_meta)

@app.route('/admin')
def admin():
    meta = meta_data()
    # destination = pd.DataFrame(result_meta,columns= ['id_tempat_rating','jenis','coordinat','tempat','kota','src_img','deskripsi'])
    # destination = destination[['tempat','jenis','kota']]
    # recent
    ad_recent = meta.tail()
    _ad_recent = ad_recent.values.tolist()

    data = meta[['id_tempat_rating','jenis','coordinat','tempat','kota','src_img','deskripsi']]
    # by trend
    # new_data = pd.merge(data,trend,on='tempat',how='left')
    # new_data['avg_rating'] = new_data['avg_rating'].fillna(0)
    # new_data['avg_rating'] = new_data['avg_rating'].round(1)
    # new_data = new_data.dropna(subset=['jenis'])
    ad_trend = meta.sort_values(by='count_tempat',ascending=False).head(10)
    _ad_trend = ad_trend.values.tolist()

    # by rating
    ad_rating = meta.sort_values(by='avg_rating',ascending=False).head(10)
    _ad_rating = ad_rating.values.tolist()
    return render_template('admin/index.html',ad_recent = _ad_recent,ad_trend=_ad_trend,ad_rating=_ad_rating)

@app.route('/admin/destination')
def admin_place():
    destination = pd.DataFrame(result_meta,columns= ['id_tempat_rating','jenis','coordinat','tempat','kota','src_img','deskripsi'])
    destination = destination[['tempat','jenis','kota']]
    _destination = destination.values.tolist()

    return render_template('admin/destination.html', destination=_destination)
@app.route('/admin/destination/<place>')
def admin_select(place=None):
    meta = meta_data()
    place = place.title()
    select_wisata = meta.loc[meta['tempat']==place]
    _destination = select_wisata.values.tolist()
    return render_template('admin/detail.html', destination=_destination)

@app.route('/admin/add_destination')
def new_destination():
    return render_template('admin/add_destination.html')

@app.route('/admin/add_destination/new', methods=["POST","GET"])
def submit_destination():
    if request.method == 'POST':
        id_tempat = request.form['id']
        name = request.form['name']
        type_destination = request.form['type']
        link_img = request.form['img']
        description = request.form['desc']
        latitude = request.form['lat']
        longitude = request.form['lon']
        city = request.form['city']
     
        con = sqlite3.connect('../../tugas-akhir.db')
        cursor = con.cursor()
        cursor.execute('INSERT INTO data(tempat,jenis,img,coo1,coo2,city,description,id_tempat) values (?,?,?,?,?,?,?,?)', (name, type_destination, link_img, latitude, longitude,city, description, id_tempat))
        con.commit()
        coordinate = latitude+","+longitude
        # sql_meta = "SELECT id_tempat,jenis, coordinat, tempat,city,img, description FROM data"
        add_places = id_tempat, type_destination, coordinate,name, city, link_img, description, 
        result_meta.append(add_places)
        # df2 = {'Name': 'Amy', 'Maths': 89, 'Science': 93}
        # df = df.append(df2, ignore_index = True)
        return redirect(url_for('admin'))

@app.route('/signin')
def signin():
    return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('daftar.html')

@app.route('/register', methods=["POST", "GET"])
def register():
    if request.method == 'POST':
        id_user = request.form['id_user']
        full_name = request.form['full_name']
        password = request.form['password']

        # print((id_user),full_name,password)

        if check_user(id_user, password):
            msg = ['A user with that username already exists.']
            return render_template('daftar.html',msg=msg)
        else:
            con = sqlite3.connect('../../tugas-akhir.db')
            cursor = con.cursor()
            cursor.execute('INSERT INTO user(nama_pengguna,id_user,password,points,level,img_src) values (?,?,?,?,?,?)', (full_name,id_user, password,0,0,'default'))
            con.commit()
            con.close()
            session['id_user'] = id_user
            return redirect(url_for('profile'))
            
@app.route('/login', methods=["POST", "GET"])
def login():
    if request.method == 'POST':
        id_user = request.form['id_user']
        password = request.form['password']
        if check_user(id_user, password):
            session['id_user'] = id_user
            return redirect(url_for('home'))
        else:
            msg = ['Sorry, your password was incorrect. Please double-check your password.']
            return render_template('login.html',msg=msg)
@app.route('/user')
def profile():
    if 'id_user' in session:
        userId = session["id_user"]
        cursor = koneksi()
        
        cursor.execute('SELECT * FROM user WHERE id_user=? order by id_user limit 1',(userId,))
        user = cursor.fetchall()

        cursor.execute('SELECT * FROM ratings WHERE id=?',(userId,))
        history = cursor.fetchall()
        n_history = len(history)
        # print(user)
        
        return render_template('profile.html' , profil=user, history=history,n_history=n_history)
    else:
        return redirect(url_for('signin'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('signin'))

@app.route('/ulas', methods=["POST", "GET"])
def ulasan():
    from datetime import date

    if request.method == 'POST':
        userId = session["id_user"]

        tempat = request.form['tempat']
        tempatId = request.form["id_tempat"]
        bintang = request.form['rating']
        review = request.form['ulasan']
        tanggal = date.today()  

        con = sqlite3.connect('../../tugas-akhir.db')
        cursor = con.cursor()
        cursor.execute('SELECT * FROM user WHERE id_user=? order by id_user limit 1',(userId,))
        user = cursor.fetchall()
        for row in user:
            nama = row[2]

        addbintang = float(bintang)
        meta['avg_rating'] = (meta['avg_rating'] * meta['count_tempat'] + addbintang)/(meta['count_tempat']+1)
        # print("AVG :",meta['avg_rating'])
        meta['count_tempat'] = meta['count_tempat'] +1
        meta['avg_rating'] = meta['avg_rating'].round(1)
        try:
            cursor.execute('INSERT INTO ratings (id,nama_pengguna,rating,tempatR,tanggal,review,id_tempat) VALUES (?,?,?,?,?,?,?)',(userId,nama,bintang,tempat,tanggal,review,tempatId))
            con.commit()
        except:
            pass
    
        return redirect(url_for('profile'))
    else:
        return redirect(url_for('signin'))

@app.route('/recs/<tempat>')
def rekomendasi(tempat):
    meta = meta_data()
    meta['features'] = meta['jenis']+meta['deskripsi']
    # meta['features'] = meta['jenis']
    #proses tfid
    tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
    # tfidf_matrix = tf.fit_transform(meta['features'])
    tfidf_matrix = tf.fit_transform(meta['features'].values.astype('U'))
    # # print(tfidf_matrix.shape)

    #cosine similarity
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    meta = meta.reset_index()
    tempats = meta['tempat']
    indices = pd.Series(meta.index, index=meta['tempat'])
    conn = sqlite3.connect('../../tugas-akhir.db')
    cursor = conn.cursor()
    # collaborative
    sql_ratings = "SELECT id, id_tempat,rating,review,tempatR,nama_pengguna FROM ratings"
    cursor.execute(sql_ratings)
    result_rating = cursor.fetchall()

    reader = Reader()

    src_rating = pd.DataFrame(result_rating,columns= ['userId','id_tempat_rating','rating','review','tempat','nama'])
    src_rating['rating'] = src_rating['rating'].astype('float')
    # src_rating['id_tempat_rating'] = src_rating['id_tempat_rating'].astype('float')
    src_rating['id_tempat_rating'] = src_rating['id_tempat_rating'].apply(convert_int)
    # print(src_rating.info())
    xidr = src_rating['id_tempat_rating']
    xidu = src_rating['userId']
    src_rating['userId'] = le.fit_transform(np.ravel(xidu))
    src_rating['id_tempat_rating'] = le.fit_transform(np.ravel(xidr))

    # src_rating['id_tempat_rating'] = src_rating['id_tempat_rating'].astype('float')
    # print(src_rating.tail())

    data_ratings = Dataset.load_from_df(src_rating[['userId','id_tempat_rating','rating']], reader)


    algo = SVD()
    svd_model = SVD(n_factors= 50, n_epochs= 30, lr_all=0.01, reg_all=0.02)

    # cross_validate(svd_model, data_ratings, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    # trainset = data_ratings.build_full_trainset()
    trainset, testset = train_test_split(data_ratings, test_size=0.20)
    svd_model.fit(trainset)
    # svd_predictions = svd_model.test(testset)

    # hybrid filtering

    store = meta[['id_tempat_rating','id_tempat']]
    # # # print(store.head(2))
    # store['id_tempat_rating'] = store['id_tempat_rating'].astype('float')

    store.columns = ['id_tempat_rating','id_tempat']
    store = store.merge(meta[['tempat', 'id_tempat_rating']], on='id_tempat_rating').set_index('tempat')
    # # # print(store.info())

    store = store.dropna()

    indices_map = store.set_index('id_tempat')
    tempat = tempat.title()
    if 'id_user' in session:
        user_id = session["id_user"]
        userId = session["id_user"]
        cursor = koneksi()
        cursor.execute('SELECT id_user,img_src FROM user WHERE id_user=? order by id_user limit 1',(user_id,))
        user = cursor.fetchall()
    else:
        user = ''

    idx = indices[tempat]
    # id_tempat = store.loc[tempat]['id_tempat']
    # print('IDX = ',idx,'\n ID = ',userId,"\n Name Place = ",tempat)
    
    wisata_id = store.loc[tempat]['id_tempat_rating']
    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:40]

    wisata_indices = [i[0] for i in sim_scores]
    # print('wi = ',wisata_indices)
    wisatas = meta.iloc[wisata_indices][['tempat', 'features','id_tempat','id_tempat_rating','jenis','kota','avg_rating','src_img','deskripsi']]
    wisatas = wisatas.merge(cities, how="left", on=["tempat"])
    wisatas = wisatas.sort_values(by='Distance', ascending=True)
    wisatas = wisatas.head(20)
    # print(wisatas.head(15))
    try:
        wisatas['est'] = wisatas['id_tempat'].apply(lambda x: svd_model.predict(userId, x).est)
        wisatas = wisatas.sort_values('est', ascending=False)
        print('HF active')
        # print(wisatas.head(15))
    except:
        print('Non-HF active')
        # pass
    wisatas = wisatas.head(10)
    # print(wisatas.info())    
    select_wisata = meta.loc[meta['tempat']==tempat]
    
    # print(select_wisata.info())
    src_rating['tempat'] = src_rating['tempat'].str.title() # nama tempat
    src_reviews = src_rating.loc[src_rating['tempat'] == tempat]
    
    
    search = meta[['tempat']]
    recS = wisatas.values.tolist()
    # print(select_wisata.info())
    arr_select_wisata = select_wisata.values.tolist()
    arr_meta = search.values.tolist()
    arr_src_reviews = src_reviews.values.tolist()
    # try:
    #     arr_select_wisata[0][10] = int(arr_select_wisata[0][10])
    # except:
    #     pass
    # arr_select_wisata[0][10] = nan
    
    kota = arr_select_wisata[0][5]
    conn = sqlite3.connect('../../tugas-akhir.db')
    cursor = conn.cursor()
    cursor.execute('Select tempat,img,type,address,latitude,longitude,city_nat FROM support_place')
    fetch_halal = cursor.fetchall()
    halal = pd.DataFrame(fetch_halal,columns= ['tempat','img','type','address','latitude','longitude','city'])
    halal = halal.dropna(subset=['latitude','longitude'])
    # print(halal.info())

    src_lat = arr_select_wisata[0][3]
    src_coordinates = [item.strip() for item in src_lat.split(',')]
    src_latitude = float(src_coordinates[0])
    src_longitude = float(src_coordinates[1])
    # print("Lan =",src_latitude,"lon = ",src_longitude)
    # print("TUPE =",type(src_latitude))
    
    start_lat_h, start_lon_h = src_latitude, src_longitude
    
    # halal['tempat'] = halal['tempat'].str.title()
    hdistances_km = []
    for row in halal.itertuples(index=False):
        hdistances_km.append(
            haversine_distance(start_lat_h, start_lon_h, row.latitude, row.longitude)
        )
    halal['distance'] = hdistances_km
    halal = halal.sort_values(by='distance', ascending=True)
    
    # print(halal.info())
    arr_halal = halal.values.tolist()
    mosque =[]
    resto =[]
    atm =[]
    hotel =[]
    for i in arr_halal:
        # category
        if i[2] == 'Mosque':
            mosque.append(i)
        elif i[2] == 'Restaurants':
            resto.append(i)
        elif i[2] == 'ATM':
            atm.append(i)
        else:
            hotel.append(i)
    
    mosque = mosque[:5]
    resto = resto[:5]
    atm = atm[:5]
    hotel = hotel[:5]

    arr_category = arr_select_wisata[0][2]
    # print(arr_category)
    title = arr_select_wisata[0][4]

    return render_template('detail.html', title=title,data=recS, _select_wisata=arr_select_wisata,_meta=arr_meta, reviews=arr_src_reviews,mosque=mosque,resto=resto,hotel=hotel,atm=atm,category=arr_category,user=user)

@app.route("/destinasi/")
@app.route("/destinasi/<filter>")
def explore(filter = None):
    meta = meta_data()
    locale = cities.sort_values(by='Distance', ascending=True)
    data = locale.merge(meta,how="left", on=["tempat"])
    # new_data = pd.merge(data,trend,on='tempat',how='left')
    # new_data['avg_rating'] = new_data['avg_rating'].fillna(0)
    # new_data['avg_rating'] = new_data['avg_rating'].round(1)
    new_data = data.dropna(subset=['jenis'])
    # print(new_data.info())
    # meta = pd.merge(meta,trend,on='tempat',how='left')
    # meta['avg_rating'] = meta['avg_rating'].fillna(0)
    # meta['avg_rating'] = meta['avg_rating'].round(1)
    _data = new_data.values.tolist()

    if 'id_user' in session:
        user_id = session["id_user"]
        cursor = koneksi()
        cursor.execute('SELECT id_user,img_src FROM user WHERE id_user=? order by id_user limit 1',(user_id,))
        user = cursor.fetchall()
    else:
        user = ''
    if (filter=='populer'):
        data_trend = new_data.sort_values(by='count_tempat',ascending=False)
        # data_trend['count_tempat'] = data_trend['count_tempat'].astype(int)
        _data = data_trend.values.tolist()
        return render_template('destinasi.html', data=_data, filter=filter, user=user)

    elif (filter == 'rating'):
        data_rate = new_data.sort_values(by='avg_rating',ascending=False)
        # print(data_rate['avg_rating'].head())
        _data = data_rate.values.tolist()
        # print(data_rate.info())
        return render_template('destinasi.html', data=_data,filter=filter, user=user)
    else:
        filter = 'terdekat'
    
    return render_template('destinasi.html', data=_data,filter=filter, user=user)

if __name__ == "__main__":
    app.run(debug=True)
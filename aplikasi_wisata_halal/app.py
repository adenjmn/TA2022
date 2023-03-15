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
# sql_meta = "SELECT id_tempat,jenis, coordinat, tempat,city,img, description FROM data"
# cursor.execute(sql_meta)
# result_meta = cursor.fetchall()
def start_data():
    cursor = koneksi()
    sql_meta = "SELECT id_tempat,jenis, coordinat, tempat,city,img, description FROM data"
    cursor.execute(sql_meta)
    result_meta = cursor.fetchall()
    return result_meta

def metadata():
    result_meta = start_data()
    # content-based
    meta = pd.DataFrame(result_meta,columns= ['id_tempat','jenis','coordinat','tempat','kota','src_img','deskripsi'])
    meta = meta.dropna(subset=['jenis'])

    # xid_city = meta['kota']

    conn = sqlite3.connect('../../tugas-akhir.db')
    cursor = conn.cursor()

    sql_trend = "SELECT tempatR, COUNT(*), AVG(rating) FROM ratings GROUP BY tempatR"
    cursor.execute(sql_trend)
    result_trend = cursor.fetchall()
    trend = pd.DataFrame(result_trend,columns= ['tempat','count_tempat','avg_rating'])
    trend['tempat'] = trend['tempat'].str.title()
    # meta['id_tempat'] = le.fit_transform(np.ravel(xid_city))
    # meta['id_tempat'] = le.fit_transform(meta['kota'])
    meta['id_tempat_rating'] = meta['id_tempat']
    # print(meta.info())

    meta['deskripsi'] = meta['deskripsi'].fillna(value="")

    trend['count_tempat'] = trend['count_tempat'].apply(convert_int)
    meta['tempat'] = meta['tempat'].str.title()
    # left join
    meta = pd.merge(meta,trend,on='tempat',how='left')
    meta['avg_rating'] = meta['avg_rating'].fillna(0)
    meta['avg_rating'] = meta['avg_rating'].round(1)

    return meta

def meta_halal():
    cursor = koneksi()
    cursor.execute('Select tempat,img,type,address,latitude,longitude,deskripsi FROM support_place')
    fetch_halal = cursor.fetchall()
    halal = pd.DataFrame(fetch_halal,columns= ['tempat','img','type','address','latitude','longitude','city'])
    halal = halal.dropna(subset=['latitude','longitude'])
    return halal
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

def find_distance():
    g = geocoder.ip('me')
    curr_lat = g.lat
    curr_lon = g.lng
    cursor = koneksi()
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
    return cities


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
    arr_meta = metadata().values.tolist()
    
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
    meta = metadata()
    # by recent
    ad_recent = meta.tail().sort_index(ascending=False)
    _ad_recent = ad_recent.values.tolist()

    data = meta[['id_tempat_rating','jenis','coordinat','tempat','kota','src_img','deskripsi']]
    # by trend
    ad_trend = meta.sort_values(by='count_tempat',ascending=False).head(10)
    _ad_trend = ad_trend.values.tolist()

    # by rating
    ad_rating = meta.sort_values(by='avg_rating',ascending=False).head(10)
    _ad_rating = ad_rating.values.tolist()

    return render_template('admin/index.html',ad_recent = _ad_recent,ad_trend=_ad_trend,ad_rating=_ad_rating)

@app.route('/admin/destination')
def admin_place():
    destination = metadata()
    destination = destination[['tempat','jenis','kota']]
    _destination = destination.values.tolist()

    return render_template('admin/destination/index.html', destination=_destination)
@app.route('/admin/destination/<place>')
def admin_select(place=None):
    meta = metadata()
    # place = place.title()
    select_wisata = meta.loc[meta['tempat']==place]
    _destination = select_wisata.values.tolist()
    # print(select_wisata.head())
    return render_template('admin/destination/detail.html', destination=_destination)

@app.route('/admin/add_destination')
def new_destination():
    cursor = koneksi()
    cursor.execute('SELECT DISTINCT jenis FROM data ORDER BY jenis')
    types = cursor.fetchall()
    return render_template('admin/destination/add_destination.html',types=types)

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
        # coordinate = latitude+","+longitude
        
        # add_places = id_tempat, type_destination, coordinate,name, city, link_img, description, 
        # result_meta.append(add_places)
        # df2 = {'Name': 'Amy', 'Maths': 89, 'Science': 93}
        # df = df.append(df2, ignore_index = True)
        return redirect(url_for('admin'))

@app.route('/admin/destination/<place>/update', methods=["POST","GET"])
def update_place(place=None):
    if request.method == 'POST':
        id_tempat = request.form['id']
        type_destination = request.form['type']
        city = request.form['city']
        link_img = request.form['img']
        description = request.form['desc']
        latitude = request.form['lat']
        longitude = request.form['lon']

        con = sqlite3.connect('../../tugas-akhir.db')
        cursor = con.cursor()
        # UPDATE data SET jenis = 'Smith' WHERE id_tempat = 6699;
        cursor.execute('UPDATE data SET jenis=?,img=?,coo1=?,coo2=?,city=?,description=? WHERE id_tempat=?', (type_destination, link_img, latitude, longitude,city, description,id_tempat))
        con.commit()
        con.close()

        return redirect(url_for('admin_place'))

@app.route('/admin/destination/<id_place>/delete', methods=["POST","GET"])
def delete_destination(id_place):
    if request.method == 'GET':
        con = sqlite3.connect('../../tugas-akhir.db')
        cursor = con.cursor()
        cursor.execute('DELETE FROM	data WHERE id_tempat = ?', (id_place,))
        con.commit()
        con.close()
        return redirect(url_for('admin'))

@app.route('/admin/place')
def halalan():
    halal = meta_halal()
    _halal = halal.values.tolist()
    return render_template('admin/halal/index.html', halal=_halal)

@app.route('/admin/place/<place>')
def admin_select_halal(place=None):
    halal = meta_halal()
    select_halal = halal.loc[halal['tempat']==place]
    _place = select_halal.values.tolist()
    return render_template('admin/halal/detail.html', place=_place)

@app.route('/admin/add_place')
def new_place():
    return render_template('admin/halal/add.html')

@app.route('/admin/add_place/new', methods=["POST","GET"])
def submit_place():
    if request.method == 'POST':
        id_tempat = request.form['id']
        name = request.form['name']
        types = request.form['type']
        link_img = request.form['img']
        description = request.form['desc']
        latitude = request.form['lat']
        longitude = request.form['lon']
        address = request.form['address']
        print(types,address)
        con = sqlite3.connect('../../tugas-akhir.db')
        cursor = con.cursor()
        cursor.execute('INSERT INTO support_place(tempat,type,img,latitude,longitude,address,deskripsi) values (?,?,?,?,?,?,?)', (name, types, link_img, latitude, longitude,address, description))
        con.commit()    
        return redirect(url_for('admin'))
    else:
        print("ERROR")
        return redirect(url_for('/'))

@app.route('/admin/place/<id_place>/delete', methods=["POST","GET"])
def delete_place(id_place):
    if request.method == 'GET':
        con = sqlite3.connect('../../tugas-akhir.db')
        cursor = con.cursor()
        cursor.execute('DELETE FROM	support_place WHERE tempat = ?', (id_place,))
        con.commit()
        con.close()
        return redirect(url_for('admin'))

@app.route('/admin/member')
def member():
    cursor = koneksi()
    status = 'admin'
    cursor.execute('SELECT * FROM user WHERE status=? order by status',(status,))
    member = cursor.fetchall()
    return render_template('admin/member/index.html', member=member)

@app.route('/admin/member/<user>')
def admin_select_member(place=None):
    halal = meta_halal()
    select_halal = halal.loc[halal['tempat']==place]
    _place = select_halal.values.tolist()
    return render_template('admin/member/detail.html', place=_place)

@app.route('/admin/add_member')
def new_member():
    return render_template('admin/member/add.html')

@app.route('/admin/add_member/new', methods=["POST","GET"])
def submit_member():
    if request.method == 'POST':
        # id_tempat = request.form['id']
        # name = request.form['name']
        # types = request.form['type']
        # link_img = request.form['img']
        # description = request.form['desc']
        # latitude = request.form['lat']
        # longitude = request.form['lon']
        # address = request.form['address']
        # print(types,address)
        # con = sqlite3.connect('../../tugas-akhir.db')
        # cursor = con.cursor()
        # cursor.execute('INSERT INTO support_place(tempat,type,img,latitude,longitude,address,deskripsi) values (?,?,?,?,?,?,?)', (name, types, link_img, latitude, longitude,address, description))
        # con.commit()    
        return redirect(url_for('member'))
    else:
        print("ERROR")
        return redirect(url_for('/'))

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
            nama = row[1]
            # print(nama)

        addbintang = float(bintang)
        meta = metadata()
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
    meta = metadata()
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
    src_rating = src_rating.sort_index(ascending=False)

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
    cities = find_distance()
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
    halal = meta_halal()
    # print(halal.info())

    
    src_lat = arr_select_wisata[0][3]
    print("ini lAT=",src_lat)
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
    meta = metadata()
    cities = find_distance()
    locale = cities.sort_values(by='Distance', ascending=True)
    data = locale.merge(meta,how="left", on=["tempat"])
    new_data = data.dropna(subset=['jenis'])
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
        _data = data_trend.values.tolist()
        return render_template('destinasi.html', data=_data, filter=filter, user=user)

    elif (filter == 'rating'):
        data_rate = new_data.sort_values(by='avg_rating',ascending=False)
        _data = data_rate.values.tolist()
        return render_template('destinasi.html', data=_data,filter=filter, user=user)
    else:
        filter = 'terdekat'
    
    return render_template('destinasi.html', data=_data,filter=filter, user=user)

if __name__ == "__main__":
    app.run(debug=True)
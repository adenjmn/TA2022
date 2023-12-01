from flask import *
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from sklearn.metrics import mean_squared_error
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder
from math import *

import sqlite3
import pandas as pd
import numpy as np
import geocoder
import time

le = LabelEncoder()

def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan

conn = sqlite3.connect('../../tugas-akhir-local-new.db')
cursor = conn.cursor()

def start_data():
    cursor = koneksi()
    sql_meta = "SELECT id_tempat,jenis, latitude, longitude, tempat,city,img, description FROM data"
    cursor.execute(sql_meta)
    result_meta = cursor.fetchall()
    return result_meta

def metadata():
    result_meta = start_data()
    meta = pd.DataFrame(result_meta,columns= ['id_tempat','jenis','latitude','longitude','tempat','kota','src_img','deskripsi'])
    # meta = meta.dropna(subset=['jenis'])

    conn = sqlite3.connect('../../tugas-akhir-local-new.db')
    cursor = conn.cursor()
    sql_trend = "SELECT id_tempat, COUNT(*), AVG(rating) FROM ratings GROUP BY id_tempat"
    cursor.execute(sql_trend)
    result_trend = cursor.fetchall()
    trend = pd.DataFrame(result_trend,columns= ['id_tempat','count_tempat','avg_rating'])
    # trend['tempat'] = trend['tempat'].str.title()
    # meta['id_tempat'] = le.fit_transform(np.ravel(xid_city))
    # meta['id_tempat'] = le.fit_transform(meta['kota'])
    meta['id_tempat_rating'] = meta['id_tempat']
    # print(meta.info())

    meta['deskripsi'] = meta['deskripsi'].fillna(value="")

    trend['count_tempat'] = trend['count_tempat'].apply(convert_int)
    meta['tempat'] = meta['tempat'].str.title()
    # left join
    meta = pd.merge(meta,trend,on='id_tempat',how='left')
    meta['avg_rating'] = meta['avg_rating'].fillna(0)
    meta['avg_rating'] = meta['avg_rating'].round(1)

    return meta

def meta_halal():
    cursor = koneksi()
    cursor.execute('Select id,tempat,img,type,address,latitude,longitude,link_gmaps FROM support_place')
    fetch_halal = cursor.fetchall()
    halal = pd.DataFrame(fetch_halal,columns= ['id','tempat','img','type','address','latitude','longitude','city'])
    halal = halal.dropna(subset=['latitude','longitude'])
    return halal
# # # print(store.info())
# # # print(store.id_tempat.value_counts())
def koneksi():
    con = sqlite3.connect('../../tugas-akhir-local-new.db')
    cur = con.cursor()
    return cur
def haversine_distance(lat1, lon1, lat2, lon2):
   r = 6371
   phi1 = np.radians(lat1)
   phi2 = np.radians(lat2)
   delta_phi = np.radians(lat2 - lat1)
   delta_lambda = np.radians(lon2 - lon1)
   a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) *   np.sin(delta_lambda / 2)**2
   res = r * (2 * np.arcsin(np.sqrt(a)))
   return np.round(res, 2)

def find_distance(latitude,longitude):
    # g = geocoder.ip('me')
    curr_lat = latitude
    curr_lon = longitude
    cursor = koneksi()
    
    start_lat, start_lon = curr_lat, curr_lon
        
    cities = metadata()
    cities['tempat'] = cities['tempat'].str.title()
    distances_km = []
    for row in cities.itertuples(index=False):
        distances_km.append(
            haversine_distance(start_lat, start_lon, row.latitude, row.longitude)
        )
    cities['distance'] = distances_km
    return cities

def check_user(id_user, password):
    cursor = koneksi()
    cursor.execute('Select id_user,password,status FROM user WHERE id_user=? and password=?', (id_user, password))
    result = cursor.fetchone()
    if result:
        if result[2]=='Admin':
            status = "Admin"
            return status
        else:
            status = "User"
            return status
        # return True
    else:
        return False

def find_admin(userId):
    cursor = koneksi()
    cursor.execute('Select id_user,nama_pengguna,password,status FROM user WHERE id_user=?', (userId,))
    user = cursor.fetchall()
    if user[0][3]=='Admin':
        return user
    else:
        return False

app = Flask(__name__)
app.secret_key = "@adenjmn"

@app.route("/")
def home():    
    if 'id_user' in session:
        msg = "Berhasil masuk"
        user_id = session["id_user"]

        cursor = koneksi()
        cursor.execute('SELECT id_user,img_src,status FROM user WHERE id_user=? order by id_user limit 1',(user_id,))
        user = cursor.fetchall()
        return render_template('index.html', msg=msg,user=user)
    else:
        return render_template('index.html')

@app.route('/admin')
def admin():
    if 'id_user' in session:
        userId = session["id_user"]
        if find_admin(userId):
            user = find_admin(userId)
            meta = metadata()
            # by recent
            ad_recent = meta.tail().sort_index(ascending=False)
            _ad_recent = ad_recent.values.tolist()
            # by trend
            ad_trend = meta.sort_values(by='count_tempat',ascending=False).head(10)
            _ad_trend = ad_trend.values.tolist()
            # by rating
            ad_rating = meta.sort_values(by='avg_rating',ascending=False).head(10)
            _ad_rating = ad_rating.values.tolist()
            return render_template('admin/index.html',ad_recent = _ad_recent,ad_trend=_ad_trend,ad_rating=_ad_rating,user=user)
        else:
            return redirect(url_for('home'))
    else:
        return redirect(url_for('signin'))

@app.route('/admin/destination')
def admin_place():
    if 'id_user' in session:
        userId = session["id_user"]
        if find_admin(userId):
            user = find_admin(userId)
            destination = metadata()
            destination = destination[['id_tempat','tempat','jenis','kota']]
            _destination = destination.values.tolist()
            return render_template('admin/destination/index.html', destination=_destination,user=user)
        else:
            return redirect(url_for('home'))
    else:
        return redirect(url_for('signin'))
@app.route('/admin/destination/<id_dest>')
def admin_select(id_dest=None):
    if 'id_user' in session:
        userId = session["id_user"]
        if find_admin(userId):
            user = find_admin(userId)
            meta = metadata()
            select_wisata = meta.loc[meta['id_tempat']==int(id_dest)]
            _destination = select_wisata.values.tolist()
            return render_template('admin/destination/detail.html', destination=_destination, user=user)
        else:
            return redirect(url_for('home'))
    else:
        return redirect(url_for('signin'))
@app.route('/admin/add_destination')
def new_destination():
    if 'id_user' in session:
        userId = session["id_user"]
        if find_admin(userId):
            user = find_admin(userId)
            cursor = koneksi()
            cursor.execute('SELECT DISTINCT jenis FROM data ORDER BY jenis')
            types = cursor.fetchall()
            return render_template('admin/destination/add_destination.html',types=types,user=user)
        else:
            return redirect(url_for('home'))
    else:
        return redirect(url_for('signin'))

@app.route('/admin/add_destination/new', methods=["POST","GET"])
def submit_destination():
    if request.method == 'POST':
        id_tempat = request.form['id']
        name = request.form['name']
        types = request.form['types']
        link_img = request.form['img']
        description = request.form['desc']
        latitude = request.form['lat']
        longitude = request.form['lon']
        city = request.form['city']
        try:
            con = sqlite3.connect('../../tugas-akhir-local-new.db')
            cursor = con.cursor()
            cursor.execute('INSERT INTO data(tempat,jenis,img,latitude, longitude,city,description,id_tempat) values (?,?,?,?,?,?,?,?)', (name, types, link_img, latitude, longitude,city, description, id_tempat))
            con.commit()
            return redirect(url_for('admin_place'))
        except:
            _pesan ="Silahkan ulangi, terjadi kesalahan"
            return render_template('admin/destination/add_destination.html', pesan=_pesan)

@app.route('/admin/destination/<place>/update', methods=["POST","GET"])
def update_place(place=None):
    if 'id_user' in session:
        userId = session["id_user"]
        if find_admin(userId):
            if request.method == 'POST':
                id_tempat = request.form['id']
                type_destination = request.form['type']
                city = request.form['city']
                link_img = request.form['img']
                description = request.form['desc']
                latitude = request.form['lat']
                longitude = request.form['lon']

                con = sqlite3.connect('../../tugas-akhir-local-new.db')
                cursor = con.cursor()
                cursor.execute('UPDATE data SET jenis=?,img=?,latitude=?,longitude=?,city=?,description=? WHERE id_tempat=?', (type_destination, link_img, latitude, longitude,city, description,id_tempat))
                con.commit()
                con.close()
                return redirect(url_for('admin_place'))
        else:
            return redirect(url_for('home'))
    else:
        return redirect(url_for('signin'))

@app.route('/admin/destination/<id_place>/delete', methods=["POST","GET"])
def delete_destination(id_place):
    if request.method == 'GET':
        con = sqlite3.connect('../../tugas-akhir-local-new.db')
        cursor = con.cursor()
        cursor.execute('DELETE FROM	data WHERE id_tempat = ?', (id_place,))
        con.commit()
        cursor.execute('DELETE FROM	ratings WHERE id_tempat = ?', (id_place,))
        con.commit()
        con.close()
        return redirect(url_for('admin_place'))

@app.route('/admin/place')
def halalan():
    if 'id_user' in session:
        userId = session["id_user"]
        if find_admin(userId):
            user = find_admin(userId)
            halal = meta_halal()
            _halal = halal.values.tolist()
            return render_template('admin/halal/index.html', halal=_halal,user=user)
        else:
            return redirect(url_for('home'))
    else:
        return redirect(url_for('signin'))

@app.route('/admin/place/<id_place>')
def admin_select_halal(id_place):
    if 'id_user' in session:
        userId = session["id_user"]
        if find_admin(userId):
            user = find_admin(userId)
            halal = meta_halal()
            select_halal = halal.loc[halal['id']==int(id_place)]
            _place = select_halal.values.tolist()
            return render_template('admin/halal/detail.html', place=_place,user=user)
        else:
            return redirect(url_for('home'))
    else:
        return redirect(url_for('signin'))

@app.route('/admin/add_place')
def new_place():
    if 'id_user' in session:
        userId = session["id_user"]
        if find_admin(userId):
            user = find_admin(userId)
            return render_template('admin/halal/add.html',user=user)
        else:
            return redirect(url_for('home'))
    else:
        return redirect(url_for('signin'))

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
        try:
            con = sqlite3.connect('../../tugas-akhir-local-new.db')
            cursor = con.cursor()
            cursor.execute('INSERT INTO support_place(id,tempat,type,img,latitude,longitude,address,deskripsi) values (?,?,?,?,?,?,?,?)', (id_tempat,name, types, link_img, latitude, longitude,address, description))
            con.commit()
            return redirect(url_for('halalan'))
        except:
            _pesan ="Silahkan ulangi, terjadi kesalahan"
            return render_template('admin/halal/add.html', pesan=_pesan)
    else:
        print("ERROR")
        return redirect(url_for('/'))

@app.route('/admin/place/<id_place>/delete', methods=["POST","GET"])
def delete_place(id_place):
    if request.method == 'GET':
        con = sqlite3.connect('../../tugas-akhir-local-new.db')
        cursor = con.cursor()
        cursor.execute('DELETE FROM	support_place WHERE id = ?', (id_place,))
        con.commit()
        con.close()
        return redirect(url_for('halalan'))

@app.route('/admin/member')
def member():
    if 'id_user' in session:
        userId = session["id_user"]
        if find_admin(userId):
            user = find_admin(userId)
            cursor = koneksi()
            status = 'Admin'
            cursor.execute('SELECT * FROM user WHERE status=? order by status',(status,))
            member = cursor.fetchall()
            return render_template('admin/member/index.html', member=member,user=user)
        else:
            return redirect(url_for('home'))
    else:
        return redirect(url_for('signin'))

@app.route('/admin/member/<id>')
def admin_select_member(id=None):
    if 'id_user' in session:
        userId = session["id_user"]
        if find_admin(userId):
            user = find_admin(userId)
            cursor = koneksi()
            cursor.execute('SELECT * FROM user WHERE id_user=?',(id,))
            user = cursor.fetchall()
            return render_template('admin/member/detail.html', user=user)
        else:
            return redirect(url_for('member'))
    else:
        return redirect(url_for('signin'))

@app.route('/admin/add_member')
def new_member():
    if 'id_user' in session:
        userId = session["id_user"]
        if find_admin(userId):
            user = find_admin(userId)
            return render_template('admin/member/add.html', user=user)
        else:
            return redirect(url_for('home'))
    else:
        return redirect(url_for('signin'))

@app.route('/admin/member/<place>/update', methods=["POST","GET"])
def update_member(place=None):
    if 'id_user' in session:
        userId = session["id_user"]
        if find_admin(userId):
            if request.method == 'POST':
                name = request.form['name']
                password = request.form['password']
              
                con = sqlite3.connect('../../tugas-akhir-local-new.db')
                cursor = con.cursor()
                cursor.execute('UPDATE user SET nama_pengguna=?,password=? WHERE id_user=?', (name,password,userId))
                con.commit()
                con.close()
                return redirect(url_for('member'))
        else:
            return redirect(url_for('member'))
    else:
        return redirect(url_for('signin'))

@app.route('/admin/member/<id>/delete', methods=["POST","GET"])
def delete_member(id):
    if request.method == 'GET':
        con = sqlite3.connect('../../tugas-akhir-local-new.db')
        cursor = con.cursor()
        cursor.execute('DELETE FROM	user WHERE id_user = ?', (id,))
        con.commit()
        con.close()
        return redirect(url_for('member'))

@app.route('/admin/add_member/new', methods=["POST","GET"])
def submit_member():
    if request.method == 'POST':
        id_user = request.form['id']
        name = request.form['name']
        password = request.form['password']
        status = 'Admin'
        img_default = 'default_admin'
        con = sqlite3.connect('../../tugas-akhir-local-new.db')
        cursor = con.cursor()
        cursor.execute('INSERT INTO user(id_user, nama_pengguna,status,img_src, password) values (?,?,?,?,?)', (id_user,name,status,img_default,password))
        con.commit()    
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
        status = 'user'

        try:
            con = sqlite3.connect('../../tugas-akhir-local-new.db')
            cursor = con.cursor()
            cursor.execute('INSERT INTO user(nama_pengguna,id_user,password,points,level,n_reviews,img_src,status) values (?,?,?,?,?,?,?,?)', (full_name,id_user, password,0,0,0,'default',status))
            con.commit()
            con.close()
            session['id_user'] = id_user
            return redirect(url_for('home'))
        except:
            msg = ['Pengguna dengan ID tersebut sudah ada.']
            return render_template('daftar.html',msg=msg)
            
@app.route('/login', methods=["POST", "GET"])
def login():
    if request.method == 'POST':
        id_user = request.form['id_user']
        password = request.form['password']
        print(check_user(id_user, password))
        if check_user(id_user, password)=='User':
            session['id_user'] = id_user
            return redirect(url_for('home'))
        elif check_user(id_user, password)=='Admin':
            session['id_user'] = id_user
            return redirect(url_for('admin'))
        else:
            msg = ['Maaf, id user dan password Anda salah. Silakan periksa kembali id user dan password Anda.']
            return render_template('login.html',msg=msg)
@app.route('/user')
def history():
    if 'id_user' in session:
        userId = session["id_user"]
        cursor = koneksi()
        cursor.execute('SELECT * FROM user WHERE id_user=? order by id_user limit 1',(userId,))
        user = cursor.fetchall()

        cursor.execute('SELECT * FROM ratings WHERE id_user=?',(userId,))
        history = cursor.fetchall()
        n_history = len(history)
        # print(user)
        
        return render_template('history.html' , profil=user, history=history,n_history=n_history)
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
        userId = session['id_user']
        tempat = request.form['tempat']
        tempatId = request.form['id_tempat']
        bintang = request.form['rating']
        review = request.form['ulasan']
        tanggal = date.today()  

        con = sqlite3.connect('../../tugas-akhir-local-new.db')
        cursor = con.cursor()
        cursor.execute('SELECT * FROM user WHERE id_user=? order by id_user limit 1',(userId,))
        user = cursor.fetchall()
        for row in user:
            nama = row[1]

        addbintang = int(bintang)
        # meta = metadata()
        # meta['avg_rating'] = (meta['avg_rating'] * meta['count_tempat'] + addbintang)/(meta['count_tempat']+1)
        # # print("AVG :",meta['avg_rating'])
        # meta['count_tempat'] = meta['count_tempat'] +1
        # meta['avg_rating'] = meta['avg_rating'].round(1)
        try:
            cursor.execute('INSERT INTO ratings (id_user,nama_pengguna,rating,tempatR,tanggal,review,id_tempat) VALUES (?,?,?,?,?,?,?)',(userId,nama,bintang,tempat,tanggal,review,tempatId))
            con.commit()
        except:
            pass
    
        return redirect(url_for('history'))
    else:
        return redirect(url_for('signin'))

@app.route('/recs/<tempat>')
def rekomendasi(tempat):
    start_time = time.time()
    meta = metadata()
    meta = meta.reset_index()
    tempats = meta['tempat']
    indices = pd.Series(meta.index, index=meta['tempat'])
    meta['features'] = meta['jenis']
    meta['features'] = meta['features'].str.lower()
    # meta['features'] = meta['features'].str.replace(r'[^\w\s]+', '')
    # meta['features'] = meta['features'].str.replace('[#,@,&]', '')
    # meta['features'] = meta['features'].str.replace('\d+', '')
    # meta['features'] = meta['features'].str.strip()
    #proses tfid
    tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words=None)
    try:
        tfidf_matrix = tf.fit_transform(meta['features'])
    except:
        tfidf_matrix = tf.fit_transform(meta['features'].values.astype('U'))

    #cosine similarity
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    conn = sqlite3.connect('../../tugas-akhir-local-new.db')
    cursor = conn.cursor()
    # collaborative
    sql_ratings = "SELECT id_user, id_tempat,rating,review,tempatR,nama_pengguna FROM ratings"
    cursor.execute(sql_ratings)
    result_rating = cursor.fetchall()

    reader = Reader()

    src_rating = pd.DataFrame(result_rating,columns= ['userId','id_tempat_rating','rating','review','tempat','nama'])
    src_rating['rating'] = src_rating['rating'].astype('float')
    src_rating['id_tempat_rating'] = src_rating['id_tempat_rating'].apply(convert_int)
    xidr = src_rating['id_tempat_rating']
    xidu = src_rating['userId']
    src_rating['userId'] = le.fit_transform(np.ravel(xidu))
    src_rating['id_tempat_rating'] = le.fit_transform(np.ravel(xidr))
    src_rating = src_rating.sort_index(ascending=False)
    data_ratings = Dataset.load_from_df(src_rating[['userId','id_tempat_rating','rating']], reader)
    algo = SVD()
    svd_model = SVD(n_factors= 50, n_epochs= 30, lr_all=0.01, reg_all=0.02)
    trainset, testset = train_test_split(data_ratings, test_size=0.20)
    svd_model.fit(trainset)
    # svd_predictions = svd_model.test(testset)

    # hybrid filtering
    store = meta[['id_tempat_rating','id_tempat']]
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
        cursor.execute('SELECT id_user,img_src,status FROM user WHERE id_user=? order by id_user limit 1',(user_id,))
        user = cursor.fetchall()
    else:
        user = ''

    idx = indices[tempat]
    select_wisata = meta.loc[meta['tempat']==tempat]
    arr_select_wisata = select_wisata.values.tolist()
        
    # wisata_id = store.loc[tempat]['id_tempat_rating']
    src_lat = float(arr_select_wisata[0][3])
    src_long = float(arr_select_wisata[0][4])
    hdistances_km = []
    for row in meta.itertuples(index=False):
        hdistances_km.append(
            haversine_distance(src_lat, src_long, row.latitude, row.longitude)
        )
    meta['distance'] = hdistances_km
    # meta = meta[meta.tempat != tempat]
    # meta = meta.sort_values(by='distance', ascending=True)
    meta['distance'] = meta['distance'].round(2)
    
    if 'id_user' in session:
        # CF
        wisatas = meta[['tempat', 'features','id_tempat','id_tempat_rating','jenis','kota','avg_rating','src_img','deskripsi','latitude','longitude','distance']]
        wisatas['est'] = wisatas['id_tempat'].apply(lambda x: svd_model.predict(userId, x).est)
        wisatas = wisatas[wisatas.tempat != tempat]
        wisatas = wisatas.sort_values('est', ascending=False)
        print('SWITCH - COLLABORATIVE')
        # Evaluate CF
        cross_validate(svd_model, data_ratings, measures=['RMSE'], cv=5, verbose=True)

    else:
        # CBF
        sim_scores = list(enumerate(cosine_sim[int(idx)]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:30]

        wisata_indices = [i[0] for i in sim_scores]
        wisatas = meta.iloc[wisata_indices][['tempat', 'features','id_tempat','id_tempat_rating','jenis','kota','avg_rating','src_img','deskripsi','latitude','longitude','distance']]
        wisatas = wisatas[wisatas.tempat != tempat]
        print('SWITCH - CONTENT BASED')
        # Evaluate CBF
        evaluate_cbf(tempat,cosine_sim,idx)

    wisatas = wisatas.head(10)
    # print(wisatas)
    
    src_rating['tempat'] = src_rating['tempat'].str.title() # nama tempat
    src_reviews = src_rating.loc[src_rating['tempat'] == tempat]
    
    search = meta[['tempat']]
    recS = wisatas.values.tolist()
    
    # arr_meta = search.values.tolist()
    arr_src_reviews = src_reviews.values.tolist()
    
    halal = meta_halal()
    # print(halal.info())

    src_latitude = float(arr_select_wisata[0][3])
    src_longitude = float(arr_select_wisata[0][4])
    
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
        if i[3] == 'Masjid':
            mosque.append(i)
        elif i[3] == 'Restoran':
            resto.append(i)
        elif i[3] == 'ATM':
            atm.append(i)
        else:
            hotel.append(i)
    
    mosque = mosque[:5]
    resto = resto[:5]
    atm = atm[:5]
    hotel = hotel[:5]

    arr_category = arr_select_wisata[0][2]
    title = arr_select_wisata[0][5]

    # Waktu selesai
    end_time = time.time()

    # Waktu pemrosesan total
    processing_time = end_time - start_time

    print(f"Waktu pemrosesan: {processing_time} detik")

    return render_template('detail.html', title=title,data=recS, _select_wisata=arr_select_wisata, reviews=arr_src_reviews,mosque=mosque,resto=resto,hotel=hotel,atm=atm,category=arr_category,user=user)

@app.route("/destinasi/")
@app.route("/destinasi/<filter>")
def explore(filter = None):
    new_data = metadata()
    
    # cities = find_distance()
    # locale = cities.sort_values(by='distance', ascending=True)
    # data = locale.merge(meta,how="left", on=["tempat"])
    # new_data = meta.dropna(subset=['jenis'])
    new_data = new_data.sort_index(ascending=True)
    # _data = new_data.values.tolist()

    if 'id_user' in session:
        user_id = session["id_user"]
        cursor = koneksi()
        cursor.execute('SELECT id_user,img_src,status FROM user WHERE id_user=? order by id_user limit 1',(user_id,))
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
    elif (filter == 'terdekat'):
        if 'location_data' in session:
            myLat = session['location_data']['latitude']
            myLong = session['location_data']['longitude']
            cities = find_distance(myLat,myLong)
        try:
            new_data = cities.sort_values(by='distance',ascending=True)
        except:
            new_data = new_data.sort_index(ascending=True)
        _data = new_data.values.tolist()
    else:
        _data = new_data.values.tolist()
    
    return render_template('destinasi.html', data=_data,filter=filter, user=user)

@app.route('/deteksi_lokasi', methods=['POST'])
def deteksi_lokasi():
    geolocation = request.json  # Menerima data JSON dari permintaan POST
    print("Received location data from JavaScript:", geolocation)
    # Lakukan operasi atau kirim respons ke JavaScript
    response_data = {"message": "Location data received successfully!"}
    latitude = geolocation['latitude']
    longitude = geolocation['longitude']
    session['location_data'] = geolocation
    # print("SESSION",session['location_data']['latitude'])
    
    return jsonify(response_data)

@app.route("/cari", methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        meta = metadata()
        keyword = request.form['search_query']
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(meta['tempat'])
        search_vector = vectorizer.transform([keyword])
        cosine_similarities = cosine_similarity(search_vector, tfidf_matrix)
        
        if 'location_data' in session:
            myLat = session['location_data']['latitude']
            myLong = session['location_data']['longitude']
            dum_meta = find_distance(myLat,myLong)
            meta['distance'] = dum_meta['distance']
            meta['cs'] = cosine_similarities[0]
            meta = meta.sort_values(by='cs',ascending=False).head()
            meta = meta.sort_values(by='distance',ascending=True)
        
        _data = meta.values.tolist()
        # print(meta.info())
        if 'id_user' in session:
            msg = "Berhasil masuk"
            user_id = session["id_user"]

            cursor = koneksi()
            cursor.execute('SELECT id_user,img_src,status FROM user WHERE id_user=? order by id_user limit 1',(user_id,))
            user = cursor.fetchall()
            return render_template('search.html', data=_data, user=user, keyword=keyword)
        else:
            return render_template('search.html', data=_data, keyword=keyword)
    return render_template('index.html')
def evaluate_cbf(title,cosine_sim,idx):
    smd = metadata()
    scores_sim = list(enumerate(cosine_sim[idx]))
    scores_sim = sorted(scores_sim, key=lambda x: x[1], reverse=True)
    user_preferences = smd['avg_rating'].values

    # Menghitung skor prediksi berdasarkan kemiripan kosinus
    predicted_scores = np.dot(cosine_sim, user_preferences) / np.sum(cosine_sim, axis=1)

    # Skor sebenarnya yang akan dibandingkan dengan skor prediksi
    actual_scores = smd['avg_rating'].values

    # Menghitung RMSE
    rmse_cbf = np.sqrt(mean_squared_error(actual_scores, predicted_scores))

    return print("Evaluating RMSE of algorithm cosine similarity",rmse_cbf)

if __name__ == "__main__":
    app.run(debug=True)
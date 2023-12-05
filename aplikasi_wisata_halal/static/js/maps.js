var map = L.map('map').setView([-6.9217, 107.6071], 10);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; OpenStreetMap contributors'
    }).addTo(map);

    // Tambahkan peta OSM sebagai layer
    var canvasIcon = L.divIcon({
      className: 'canvas-marker',
      iconSize: [30, 30],
      iconAnchor: [15, 30],
      html: '<canvas width="30" height="30"></canvas>'
    });

    var canvasMarker = L.marker([0, 0], { icon: canvasIcon }).addTo(map);

    // Ambil konteks elemen canvas
    var canvas = canvasMarker._icon.querySelector('canvas');
    var context = canvas.getContext('2d');

    // Gambar bentuk pin pada elemen canvas
    context.beginPath();
    context.moveTo(15, 0);
    context.lineTo(30, 30);
    context.lineTo(0, 30);
    context.fillStyle = 'purple';
    context.fill();

    // Fungsi untuk menanggapi klik pada peta
    function onMapClick(e) {
      var clickedLatLng = e.latlng;

      // Atur posisi marker pada koordinat yang diklik
      canvasMarker.setLatLng(clickedLatLng);

      // Tampilkan koordinat yang dipilih
      document.getElementById('info_lat').textContent = clickedLatLng.lat.toFixed(10);
      document.getElementById('info_lon').textContent = clickedLatLng.lng.toFixed(10);
      let lat = document.querySelector('input[name="lat"]');
      let lon = document.querySelector('input[name="lon"]');
      lat.value = clickedLatLng.lat.toFixed(10);
      lon.value = clickedLatLng.lng.toFixed(10);
    }

    // Tambahkan event click pada peta
    map.on('click', onMapClick);
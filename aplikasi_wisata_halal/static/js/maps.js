var geocoder = new google.maps.Geocoder();

function geocodePosition(pos) {
    geocoder.geocode({
        latLng: pos
    }, function (responses) {
        if (responses && responses.length > 0) {
            updateMarkerAddress(responses[0].formatted_address);
        } else {
            updateMarkerAddress('Cannot determine address at this location.');
        }
    });
}

function updateMarkerStatus(str) {
    document.getElementById('markerStatus').innerHTML = str;
}

function updateMarkerPosition(latLng) {
    var info_lat = document.getElementById('info_lat').innerHTML = latLng.lat();
    var info_lon = document.getElementById('info_lon').innerHTML = latLng.lng();

    let lat = document.querySelector('input[name="lat"]');
    let lon = document.querySelector('input[name="lon"]');
    lat.value = info_lat;
    lon.value = info_lon;
    // document.getElementById("demo").value = info_lat;
}

function updateMarkerAddress(str) {
    document.getElementById('address').innerHTML = str;
}

function updateMarkerURL(str) {
    var pos = [str.lat(), str.lng()].join(',');
    var link = "https://www.google.com/maps/place/@" + pos;

    var htmlLink = document.getElementById("url");
    htmlLink.innerHTML = "Open Maps";
    htmlLink.setAttribute('href', link);
}

function initialize() {
    var latLng = new google.maps.LatLng(-6.1312767, 106.8001397);
    var map = new google.maps.Map(document.getElementById('mapCanvas'), {
        zoom: 15,
        center: latLng,
        mapTypeId: google.maps.MapTypeId.ROADMAP,
        streetViewControl: false
    });

    var image = 'https://cdn1.iconfinder.com/data/icons/Map-Markers-Icons-Demo-PNG/64/Map-Marker-Marker-Outside-Chartreuse.png';

    var marker = new google.maps.Marker({
        position: latLng,
        title: 'Posisi Saya',
        map: map,
        draggable: false,
        icon: image,
    });
    // Update current position info.
    updateMarkerPosition(latLng);
    geocodePosition(latLng);

    // Add dragging event listeners.
    //           google.maps.event.addListener(marker, 'dragstart', function() {
    //             updateMarkerAddress('Dragging...');
    //           });

    //           google.maps.event.addListener(marker, 'drag', function() {
    //             updateMarkerStatus('Dragging...');
    //             updateMarkerPosition(marker.getPosition());
    //           });

    //           google.maps.event.addListener(marker, 'dragend', function() {
    //             updateMarkerStatus('Drag ended');
    //             geocodePosition(marker.getPosition());
    //           });

    map.addListener('center_changed', function () {
        marker.setPosition(map.getCenter());
        //updateMarkerStatus('Center of Map');
        geocodePosition(marker.getPosition());
        updateMarkerPosition(marker.getPosition());
        updateMarkerURL(marker.getPosition());
    });

    if (navigator.geolocation) {

        navigator.geolocation.getCurrentPosition(function (position) {
            user_location = new google.maps.LatLng(position.coords.latitude, position.coords.longitude);
            map.setCenter(user_location);
        });
    } else {
        /* Browser doesn't support Geolocation */
    }
}

// Onload handler to fire off the app.
google.maps.event.addDomListener(window, 'load', initialize);
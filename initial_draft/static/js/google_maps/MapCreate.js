export default function buildMapWindow(lat, long, zoom) {
    var mapOptions = {
        zoom: zoom,
        mapTypeId: google.maps.MapTypeId.HYBRID,
        center: new google.maps.LatLng(lat, long)
    };

    var map = new google.maps.Map(document.getElementById("map"), mapOptions);
    return map;
}
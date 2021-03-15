import buildMapWindow from './MapCreate.js';

export class MAP {
    constructor(lat, long, zoom = 13, map_holder=null, drag_listener=false) {
        this.lat = lat;
        this.long = long;
        this.info = new google.maps.InfoWindow();
        this.map = null;
        this.zoom = zoom;
        this.drawing_manager = null;
        this.overlay_shape = null;

        this.drag_listerner = drag_listener;

        if (map_holder)
            this.map_holder = map_holder;
    };

    init_map() {
        this.map = buildMapWindow(this.lat, this.long, this.zoom);
    };

    init_draw_manager(){
        this.drawing_manager = this.create_draw_manager();
        this.drawing_manager.setMap(this.map);
    };

    create_draw_manager(){
        return new google.maps.drawing.DrawingManager({
            drawingMode: null,
            drawingControl: true,
            drawingControlOptions: {
                position: google.maps.ControlPosition.TOP_CENTER,
                drawingModes: ['rectangle'],
            },
            rectangleOptions:{
                editable:true,
                draggable:true,
            }
        });
    };

    add_listeners(){
        var map = this;
        google.maps.event.addListener(this.drawing_manager, 'overlaycomplete', function(e) {
            var bounds = e.overlay.getBounds();
            var start = bounds.getNorthEast();
            var end = bounds.getSouthWest();
            if( map.map_holder ){
                map.map_holder.bounds = {
                    "NE": {
                        "lat": start.lat(),
                        "long": start.lng(),
                    },
                    "SW":{
                        "lat": end.lat(),
                        "long": end.lng(),
                    }
                }
            }

            if (this.drag_listerner){
                add_drag_listener();
            }

            map.drawing_manager.setOptions({
                drawingControl: false
            });
            map.drawing_manager.setDrawingMode(null);

        });

        function add_drag_listener(){
            map.overlay_shape.addListener(map.drawing_manager, 'dragend', function(e) {
                console.log(e.overlay.getBounds());
            });
        }
    };

    add_custom_marker( lat, long, image = null ){
        let custom_marker;

        if(image){
            var icon = {
                url: image, // url
                scaledSize: new google.maps.Size(20, 20), // scaled size
                origin: new google.maps.Point(0,0), // origin
                anchor: new google.maps.Point(10,10) // anchor
            };

             custom_marker = new google.maps.Marker({
                position: { lat: lat, lng: long },
                map:this.map,
                icon: icon,
            });
        }else{
            custom_marker = new google.maps.Marker({
                position: { lat: lat, lng: long },
                map:this.map,
            });
        }

        return custom_marker;
    }
}
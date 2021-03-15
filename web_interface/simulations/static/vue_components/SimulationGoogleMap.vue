<template id="template">
    <div class="row">
        <div class="card">
            <div class="card-content">
                <span class="card-title"> Simulation Map </span>

                <div class="row">
                    <div class="col s12">
                        <h6 style="display: inline-block;">Current timestep : {{ current_timestep }}</h6>

                        <button v-if="game_done" data-target="results_modal" class="btn modal-trigger"
                                style="display: inline; margin-left: 10px;">
                            View Results
                        </button>
                    </div>
                </div>

                <div id="map" class="google_map" style="height: 700px;">
                </div>

                <div id="results_modal" class="modal" v-if="game_done">
                    <div class="modal-content">
                        <h4>Game Results</h4>
                        <p> Game Score : {{ game_score }} </p>
                        <p> Time played : {{ game_time_played }} </p>
                    </div>
                    <div class="modal-footer">
                        <a class="modal-close btn-flat">Close</a>
                        <a href="/simulations/create" class="modal-close btn-flat">New game</a>
                    </div>
                </div>

            </div>

        </div>

        <div class="loader-holder" v-if="!play_flag">
            <div class="loader" ></div>
        </div>

    </div>
</template>

<script>
    module.exports = {
        name: "SimulationGoogleMap",
        template:"#template",
        props:["config", "simulation"],
        data(){
            return{
                map:{},
                map_bounds:{},
                map_rectangle:null,
                map_zoom:null,
                settings:{
                    size:[]
                },
                center_lat:0,
                center_long:0,
                rectangles:[],
                cookie: this.getCookie('swarm_southampton_cookie'),
                current_timestep:0,

                socket: null,
                game_done:false,
                game_score:0,
                game_time_played:"",
                play_flag:false,
                total_timesteps: 100,
            }
        },
        methods:{
            async sleep(ms) {
              return new Promise(resolve => setTimeout(resolve, ms));
            },

            // get data

            async GET(url, timeout, nr_tries){
                let checked = false;
                let current_tries = 0;
                let data = {};

                while( !checked && current_tries < nr_tries) {
                    await fetch(url)
                        .then(response => response.json())
                        .then(json_parsed => {
                            if( (json_parsed instanceof Array && json_parsed.length > 0) ||
                                (json_parsed instanceof Object && Object.keys(json_parsed).length > 0) ){
                                    data = json_parsed;
                                    checked = true;
                            }

                        })
                        .catch(err => {
                            console.log('Request Failed', err);
                        });

                    if(!checked) {
                        await this.sleep(timeout);
                        current_tries++;
                    }
                }

                return data;
            },

            // get user cookie
            getCookie(cookie){
                const value = `; ${document.cookie}`;
                const parts = value.split(`; ${cookie}=`);
                if (parts.length === 2) return parts.pop().split(';').shift();
            },

            // send data
            SEND( data, url, headers = null ){
                var xhr = new XMLHttpRequest();
                xhr.open("POST", url, true);
                if(headers === null){
                    xhr.setRequestHeader('Content-Type', 'application/json');
                }else{
                    Object.keys(headers).forEach((key) => {
                        xhr.setRequestHeader(key, headers[key]);
                    });
                }
                xhr.send(JSON.stringify(data));
            },

            async PATCH(data, url){
                let resp = null;
                await fetch(url,{
                    method:'PATCH',
                    body:JSON.stringify(data),
                    headers: {"Content-type": "application/json; charset=UTF-8"},
                }).then(response => response.json())
                .then(json_parsed => {
                    if( (json_parsed instanceof Array && json_parsed.length > 0) ||
                        (json_parsed instanceof Object && Object.keys(json_parsed).length > 0) ){
                            resp = json_parsed;
                    }

                })
                .catch(err => {
                    console.log('Request Failed', err);
                });
                return resp;
            },

            // send operator action
            send_operator_action(action_name, row, col){
                let json_data = {
                    "id": this.simulation.id,
                    "operation": "update",
                    "action": action_name,
                    "pos": [row, col],
                };

                this.socket.send(JSON.stringify(json_data));
            },

            // send statistics data
            send_behaviour(action_name){
                let json_data = {
                    "action": action_name,
                    "time": new Date(),
                    "timestep": this.current_timestep, // need to implement once we finish integrating and revamping the play function
                    "simulation_id": this.simulation.id,
                    "cookie": this.cookie
                };

                this.SEND(json_data, `/api/v1/simulations/${this.simulation.id}/records/`)
            },

            // setup
            async importMap(){
                let url = 'http://localhost:8000/static/js/google_maps/Map.js';
                const {MAP} = await import(url);
                this.map = new MAP(this.center_lat, this.center_long, 14);
                this.map.init_map();
                this.map_zoom = this.map.map.getZoom();

                this.add_rectangle();
                this.createInitialGrid(null);
                this.add_listeners();

                this.map.map.set('disableDoubleClickZoom', true);

                await this.set_socket();
            },

            add_rectangle(){
                const rectangle = new google.maps.Rectangle({
                    strokeColor: "#ffffff",
                    strokeOpacity: 0.8,
                    strokeWeight: 2,
                    fillColor: "#ffffff",
                    fillOpacity: 0.35,
                    map:this.map.map,
                    bounds: this.map_bounds,
                });
                this.map_rectangle = rectangle;
            },

            add_listeners(){
                this.map.map.addListener("dragend", () => {
                    if(!app.game_done)
                        this.send_behaviour("dragged");
                });

                this.map.map.addListener("zoom_changed", () => {
                    if( this.map_zoom < this.map.map.getZoom() ){
                        if(!app.game_done)
                            this.send_behaviour("zoom_in");
                    }else{
                        if(!app.game_done)
                            this.send_behaviour("zoom_out");
                    }
                    this.map_zoom = this.map.map.getZoom();
                });

            },

            async set_socket(){
                let app = this;
                this.socket = new WebSocket('ws://localhost:8765');
                let data = {
                    "id": this.simulation.id,
                    "operation": "start",
                    "config":{
                        "drones": this.simulation.drones,
                        "width": this.simulation.width,
                        "height": this.simulation.height,
                    }
                };

                 this.socket.onmessage = async function (event) {
                    let resp = JSON.parse(event.data);

                     if(resp.operation === "start") {
                         app.play_flag = true;
                         app.total_timesteps = parseInt(resp.timesteps);
                         await app.play();
                     }else if(resp.operation === "close"){
                         let data = {
                            'score':resp['score'],
                         };
                         let req  = await app.PATCH(data, `/api/v1/simulations/${app.simulation.id}/`);
                         console.log(req);

                         if (req !== undefined) {
                            app.game_time_played = req['time_played'];
                            app.game_time_played = req['score'];
                         }

                     }else if(resp.operation === "get_data"){
                         // handle data update
                         console.log(resp);
                     }
                };

                this.socket.onopen = function(event){
                    app.socket.send(JSON.stringify(data));
                }
            },

            // map processing

            createInitialGrid(confidence) {
                let app = this;
                let left_dist = Math.abs(this.map_rectangle.getBounds().getNorthEast().lng() - this.map_rectangle.getBounds().getSouthWest().lng());
                let below_dist = Math.abs(this.map_rectangle.getBounds().getNorthEast().lat() - this.map_rectangle.getBounds().getSouthWest().lat());

                let sq_lat = below_dist / this.settings.size[0];
                let sq_long = left_dist / this.settings.size[1];

                for (var i = 0; i < this.settings.size[0]; i++) {
                    let map_col = [];

                    for (var j = 0; j < this.settings.size[1]; j++) {
                        const row = i;
                        const col = j;

                        let top_lat = this.map_rectangle.getBounds().getNorthEast().lat() - (sq_lat * i);
                        let btm_lat = this.map_rectangle.getBounds().getNorthEast().lat() - (sq_lat * (i + 1));

                        let left_lng = this.map_rectangle.getBounds().getSouthWest().lng() + (sq_long * j);
                        let right_lng = this.map_rectangle.getBounds().getSouthWest().lng() + (sq_long * (j + 1));

                        if (confidence === null)
                            overlay = 0;
                        else
                            overlay = 1 - confidence[i][j];

                        let rectangle = new google.maps.Rectangle({
                            strokeColor: '#FFFFFF',
                            strokeOpacity: 0,
                            strokeWeight: 2,
                            fillColor: '#FFFFFF',
                            fillOpacity: overlay,
                            map: app.map.map,
                            bounds: new google.maps.LatLngBounds(
                                new google.maps.LatLng(top_lat, left_lng),
                                new google.maps.LatLng(btm_lat, right_lng)
                            ),
                        });

                        rectangle.addListener("click", () => {
                            if(!app.game_done)
                                this.send_operator_action("attract", row, col);
                        });

                        rectangle.addListener("rightclick", () => {
                            if(!app.game_done)
                                this.send_operator_action("deflect", row, col);

                        });

                        map_col.push({
                            "overlay": rectangle,
                            "marker_pos": rectangle.getBounds().getCenter()
                        });
                    }

                    this.rectangles.push(map_col);
                }
            },

            draw_confidence(confidence){
                if( this.rectangles.length === 0 ){
                    this.createInitialGrid(confidence);
                }else{
                    this.rectangles.forEach( (rectangles_row, row) => {
                        rectangles_row.forEach( (rectangle, col) => {
                            let new_opac = 1 - confidence[row][col];
                            rectangle.overlay.fillOpacity = new_opac;
                            rectangle.overlay.fillColor = "white";
                            rectangle.overlay.setOptions({fillOpacity: new_opac });
                        });
                    });
                }
            },

            draw_disasters(disasters){
                disasters.forEach((disaster) => {
                    let coords = disaster[0];
                    let opacity = disaster[1];
                    this.rectangles[coords[0]][coords[1]].overlay.fillColor = "red";
                    this.rectangles[coords[0]][coords[1]].overlay.fillOpacity = opacity;
                });
            },

            // main method

            async play(){
                let req;
                for( var i = 1; i <= this.total_timesteps; i++ ){
                    if(!this.game_done) {
                        req = await this.GET(`/api/v1/simulations/${this.simulation.id}/timestep/${i}`, 100, 100);
                        this.current_timestep = i;

                        if (req[0] !== undefined) {
                            let data = JSON.parse(req[0].config.replaceAll("\'", "\""));

                            let disasters = data.belief;
                            let confidence = data.confidence;

                            this.draw_confidence(confidence);
                            this.draw_disasters(disasters);
                        }
                    }else
                        break;
                }

                this.finish_game();
            },

            finish_game(){
                // get score and time played
                this.game_done = true;

                $(document).ready(function(){
                    $('.modal').modal();
                    $('.modal').modal('open');
                });

            },
        },

        mounted(){
            this.map_bounds.north = this.config.NE.lat;
            this.map_bounds.south = this.config.SW.lat;
            this.map_bounds.east = this.config.NE.long;
            this.map_bounds.west = this.config.SW.long;

            this.center_lat = (this.map_bounds.north + this.map_bounds.south) / 2;
            this.center_long = (this.map_bounds.east + this.map_bounds.west) / 2;

            this.settings.size = [this.simulation.width, this.simulation.height];

            this.importMap();
        }
    }
</script>
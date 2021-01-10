<template id="template">
    <div class="card">
        <div class="card-content">
            <span class="card-title"> Simulation Map </span>

            <input type="hidden" name="borders" :value="JSON.stringify(bounds)">

            <div id="map" class="google_map">
            </div>
        </div>
    </div>
</template>

<script>
    // import {MAP} from ;

    module.exports = {
        name: "SimulationMapSize",
        template:"#template",
        data(){
            return{
                map:{},
                bounds:{}
            }
        },
        methods:{
            async importMap(){
                let url = 'http://localhost:8000/static/js/google_maps/Map.js';
                const {MAP} = await import(url);
                this.map = new MAP(50.9097, -1.4044, 13, this, true);
                this.map.init_map();
                this.map.init_draw_manager();
                this.map.add_listeners();
            }
        },
        mounted(){
            this.importMap();
        }
    }
</script>
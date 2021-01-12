<template id="template">
    <div class="my-content">
        <div class="row">
            <div class="col s12">
                <simulation-google-map :config="config" :simulation="simulation"></simulation-google-map>
            </div>
        </div>
    </div>
</template>

<script>
    module.exports = {
        name: "ViewSimulation",
        template:"#template",
        data(){
            return{
                simulation:{},
                config:{},
            }
        },
        mounted(){
            this.simulation = JSON.parse(document.getElementById('simulation').textContent);
            this.config = JSON.parse(JSON.parse(document.getElementById('config').textContent));
        },
        components:{
            'SimulationBeliefMap': 'url:/static/vue_components/SimulationBeliefMap.vue',
            'SimulationConfidenceMap': 'url:/static/vue_components/SimulationConfidenceMap.vue',
            'SimulationGoogleMap': 'url:/static/vue_components/SimulationGoogleMap.vue',
        },
    }
</script>
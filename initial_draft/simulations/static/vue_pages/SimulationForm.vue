<template id="template">
    <div class="my-content">
        <div class="row">
            <div class="col s12 m8">
                <simulation-config></simulation-config>
            </div>
        </div>

        <div class="row">
            <div class="col s12">
                <simulation-map-config></simulation-map-config>
            </div>
        </div>

        <div class="row">
            <div class="col s12">
                <div class="card">
                    <div class="card-content">
                        <button class="btn btn-block" type="submit" name="add_simulation">Save</button>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script>
    module.exports = {
        name: "SimulationForm",
        template:"#template",
        components:{
            'SimulationConfig': 'url:/static/vue_components/SimulationConfig.vue',
            'SimulationMapConfig': 'url:/static/vue_components/SimulationMapConfig.vue',
        },
    }
</script>
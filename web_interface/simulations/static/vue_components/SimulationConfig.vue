<template id="template">
    <div class="card">
        <div class="card-content">
            <span class="card-title"> Simulation parameters </span>

            <div class="row">
                <div class="input-field col s4">
                    <i class="material-icons prefix">label</i>
                    <input id="name" type="text" class="validate" name="sim_name" required v-model="name">
                    <label for="name">Simulation Name</label>
                </div>

                <div class="input-field col s4">
                    <select id="level" name="level" v-model="level_selected" required>
                        <option value="" disabled selected>Please select a diffuclty level</option>
                        <option v-for="level in levels" :value="level.value">{{ level.desc }}</option>
                    </select>
                    <label>Simulation Difficulty Level</label>
                </div>

            </div>
        </div>
    </div>
</template>

<script>
    module.exports = {
        name: "SimulationConfig",
        template:"#template",
        data(){
            return{
                name:"",
                drones:"10",
                level_selected:"",
                levels:[
                    { value : "easy", desc: "Easy ( 15 drones, 40X40 Map )" },
                    { value : "medium", desc: "Medium ( 15 drones, 40X40 Map )" },
                    { value : "hard", desc: "Hard ( 15 drones, 40X40 Map )" },
                ]
            }
        },
        mounted(){
            $(document).ready(function(){
                $('select').formSelect();
            });
        }
    }
</script>
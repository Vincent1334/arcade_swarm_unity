<template id = "template">
    <div class="my-content">
        <div class="row">
            <div class="col s12">
                <div class="card">
                    <div class="card-content">
                        <simulation-table :simulations="simulations"></simulation-table>
                    </div>
                </div>
            </div>

            <div class="col s12">
                <div class="card">
                    <div class="card-content" v-if="all_simulations.length > 0">
                        <pagination :total="all_simulations.length" :page="page" :perpage="perPage" @changepage="change_page"></pagination>
                    </div>
                </div>
            </div>

        </div>
    </div>
</template>

<script>
    module.exports = {
        name: 'Simulations',
        template:"#template",
        data : function(){
            return{
                all_simulations: [],
                simulations: [],
                perPage:10,
                page:1,
            }
        },
        methods:{
            async getData(){
                const res = await fetch('http://localhost:8000/api/v1/simulations/');
                const data = await res.json();
                this.all_simulations = data;
                this.simulations = this.all_simulations.slice(0, 10);
            },

            async updateData(){
                while(true){
                    await this.sleep(10000);
                    await this.getData();
                }
            },

            sleep(ms) {
              return new Promise(resolve => setTimeout(resolve, ms));
            },

            change_page(p){
                this.page=p;
                this.simulations = this.all_simulations.slice( this.perPage * (this.page-1), this.perPage * (this.page) )
            }
        },
        beforeMount(){
            this.getData();
        },
        mounted(){
            // this.updateData();
        },
        components:{
            'Pagination': 'url:/static/vue_components/Pagination.vue',
            'SimulationTable': 'url:/static/vue_components/SimulationsTable.vue',
        }
    }
</script>
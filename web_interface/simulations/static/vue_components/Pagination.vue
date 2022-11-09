<template id="template">

  <div class="row">
        <ul class="pagination">

            <li :class="{disabled: page == 1}"><a href="#!" v-on:click="setPage(page-1)"><i class="material-icons">chevron_left</i></a></li>

            <li v-for="p in pages" :class="{active: p == page}"><a href="#!" v-on:click="setPage(p)">{{p}}</a></li>

            <li :class="{disabled: page == pages[pages.length - 1]}"><a href="#!" v-on:click="setPage(page+1)"><i class="material-icons">chevron_right</i></a></li>
        </ul>
  </div>

</template>

<script>
    module.exports = {
        name: 'Pagination',
        template:"#template",
        props:["total", "page", "perpage"],
        data: function(){
            return{
                pages:[],
            }
        },
        methods:{
            setPages () {
                let numberOfPages = Math.ceil(this.total / this.perpage);
                for (let index = 1; index <= numberOfPages; index++) {
                    this.pages.push(index);
                }
            },
            setPage(p){
                if( p > 0 && p <= this.pages.length )
                    this.$emit('changepage', p)
            }
        },
        mounted(){
            this.setPages();
        },
    }
</script>
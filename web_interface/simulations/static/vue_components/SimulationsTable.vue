<template id="template">
    <div>
        <table class="responsive-table">
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Name</th>
                    <th>Status</th>
                    <th>Difficulty Level</th>
                    <th>Number Of Drones</th>
                    <th>Map Size</th>
                    <th>Actions</th>
                </tr>
            </thead>

            <tbody>
                <tr v-for="simulation in simulations">
                    <td>{{ simulation.id.split("-")[0] }}</td>
                    <td>{{ simulation.name }}</td>
                    <td>
                        <span class= "badge" :class="'badge-' + simulation.status_label">{{ simulation.status }}</span>
                    </td>
                    <td>
                        {{simulation.level | title}}
                    </td>
                    <td>{{ simulation.drones }}</td>
                    <td>{{ simulation.width }} X {{ simulation.height }}</td>
                    <td>
                        <a v-if="simulation.status == 'Preparing'" :href = "'/simulations/create/' + simulation.id" class="waves-effect waves-light btn">Edit Simulation</a>
                        <a v-else :href = "'/simulations/' + simulation.id" class="waves-effect waves-light btn">View Results</a>
                    </td>
                </tr>
            </tbody>
        </table>
    </div>

</template>

<script>
    module.exports = {
        name: 'SimulationsTable',
        template: "#template",
        props:["simulations"],
        filters:{
            title: (value) => {
                if (!value) return '';
                value = value.toString();
                return value.charAt(0).toUpperCase() + value.slice(1);
            }
        }
    }
</script>
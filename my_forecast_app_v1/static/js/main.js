/**
 * Inicializa el componente DataTables en la tabla de métricas
 * si esta se encuentra presente en la página.
 */
$(document).ready(function() {
    // Verificamos si la tabla de métricas fue renderizada
    if ($('#metrics-table').length) {
        // Activamos DataTable para ordenar y paginar la tabla
        $('#metrics-table').DataTable();
    }
});

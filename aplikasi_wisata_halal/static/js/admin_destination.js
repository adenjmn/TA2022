$(document).ready(function() {
    $('#example').DataTable( {
        scrollY:        300,
        scrollX:        true,
        scrollCollapse: true,
        paging:         true,
        responsive:     true,
        fixedHeader:           {
            header: true,
            footer: true
        }
    } );
} );
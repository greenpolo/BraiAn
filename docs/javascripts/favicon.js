$(document).ready(function() {
    // readthedocs: rel="shortcut icon"
    // material: rel="icon"
    var link = document.querySelector("link[rel~='icon']");
    if (!link) {
        link = document.createElement('link');
        link.rel = 'icon';
        document.head.appendChild(link);
    }
    link.href = '../resources/favicon.svg';
})

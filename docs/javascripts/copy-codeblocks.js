$(document).ready(function() {
    var selectors = document.querySelectorAll('pre code');

    var copyButton =
    '<div class="clipboard">'+
      '<button class="btn-clipboard" title="Copy to clipboard">'+
        '<div>'+
          '<span class="notice">Copied!</span>'+
          '<svg aria-hidden="true" class="clipboard-copy-icon" data-view-component="true" height="20" version="1.1" viewBox="0 0 16 16" width="20">'+
            '<path d="M0 6.75C0 5.784.784 5 1.75 5h1.5a.75.75 0 010 1.5h-1.5a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-1.5a.75.75 0 011.5 0v1.5A1.75 1.75 0 019.25 16h-7.5A1.75 1.75 0 010 14.25v-7.5z" fill="currentColor" fill-rule="evenodd"></path>'+
            '<path d="M5 1.75C5 .784 5.784 0 6.75 0h7.5C15.216 0 16 .784 16 1.75v7.5A1.75 1.75 0 0114.25 11h-7.5A1.75 1.75 0 015 9.25v-7.5zm1.75-.25a.25.25 0 00-.25.25v7.5c0 .138.112.25.25.25h7.5a.25.25 0 00.25-.25v-7.5a.25.25 0 00-.25-.25h-7.5z" fill="currentColor" fill-rule="evenodd"></path>'+
          '</svg>'+
        '</div>'+
      '</button>'+
    '</div>';

    Array.prototype.forEach.call(selectors, function(selector){
      selector.insertAdjacentHTML('beforebegin', copyButton);
    });
    var clipboard = new ClipboardJS('.btn-clipboard', {
      target: function (trigger) {
        return trigger.parentNode.nextElementSibling;
      }
    });

    clipboard.on('success', function (e) {
      e.clearSelection();

      var copiedText = e.trigger.firstChild.firstChild;
      copiedText.classList.add('show');
      setTimeout(function() { copiedText.classList.remove('show'); }, 1000);
    });
  });
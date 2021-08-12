    window.onload = function(){
      element = document.getElementById("download-button");
      value = element.getAttribute("download") ;
      if ((value == "None" )){
        element.classList.add("hidden");
      }
    }

    function downloadButtonVisibility(){
      element = document.getElementById("download-button");
      value = element.getAttribute("download") ;

      if ((value != "None" )){
        element.classList.remove("hidden");
      }
    }

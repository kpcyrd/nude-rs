
var options = {
  chart: {
    height: 350,
      type: 'bar',
    },
    plotOptions: {
      bar: {
        horizontal: true,
      }
    },
    dataLabels: {
      enabled: false
    },
    series: [{
      data: [525, 17931, 65289]
    }],
    title: {
      text: 'Classification of 4 images, 10x each'
    },
    xaxis: {
      categories: ['nude-rs', 'nude-js', 'nude.py'],
      labels: {
        formatter: function(val) {
          return val + "ms"
        }
      }
    }
}

var chart = new ApexCharts(
  document.querySelector("#chart"),
  options
);
  
chart.render();

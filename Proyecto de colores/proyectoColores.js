
function mostrarDataset() {
    const tableBody = document.getElementById("dataset-body");

    dataset.forEach(color => {
        const row = document.createElement("tr");
        row.innerHTML = `
            <td>${color.nombre}</td>
            <td>${color.r}</td>
            <td>${color.g}</td>
            <td>${color.b}</td>
        `;
        tableBody.appendChild(row);
    });
}

mostrarDataset();


// Codigo del KNN

function predictColor(k = 5) {
    const r = parseInt(document.getElementById("r").value);
    const g = parseInt(document.getElementById("g").value);
    const b = parseInt(document.getElementById("b").value);

    // Validar entrada
    if (isNaN(r) || isNaN(g) || isNaN(b) || r < 0 || r > 255 || g < 0 || g > 255 || b < 0 || b > 255) {
        alert("Por favor, ingresa valores RGB válidos entre 0 y 255.");
        return;
    }

    // Calcular las distancias de cada color en el dataset
    const distances = dataset.map(color => ({
        nombre: color.nombre,
        distance: Math.sqrt(
            Math.pow(r - color.r, 2) +
            Math.pow(g - color.g, 2) +
            Math.pow(b - color.b, 2)
        )
    }));

    // Ordenar por distancia (menor a mayor)
    distances.sort((a, b) => a.distance - b.distance);

    // Obtener los k vecinos más cercanos
    const nearestNeighbors = distances.slice(0, k);

    // Contar la frecuencia de los nombres entre los k vecinos
    const frequency = {};
    nearestNeighbors.forEach(neighbor => {
        frequency[neighbor.nombre] = (frequency[neighbor.nombre] || 0) + 1;
    });

    // Encontrar el nombre con mayor frecuencia
    let predictedColor = null;
    let maxCount = 0;
    for (const [color, count] of Object.entries(frequency)) {
        if (count > maxCount) {
            maxCount = count;
            predictedColor = color;
        }
    }

    // Mostrar el resultado
    document.getElementById("result").innerText = `Resultado: ${predictedColor || 'No se encontró un color cercano'}`;
    document.getElementById("color-box").style.backgroundColor = `rgb(${r}, ${g}, ${b})`;
}


function evaluateK(maxK) {
    const results = [];

    // Dividir el dataset en conjunto de prueba y entrenamiento
    const testDataset = dataset.map(color => ({
        nombre: color.nombre,
        r: color.r,
        g: color.g,
        b: color.b
    }));

    for (let k = 1; k <= maxK; k++) {
        let correctPredictions = 0;

        testDataset.forEach(testColor => {
            // Simular exclusión del punto de prueba
            const neighbors = dataset.filter(color => color !== testColor);

            // Calcular distancias
            const distances = neighbors.map(color => ({
                nombre: color.nombre,
                distance: Math.sqrt(
                    Math.pow(testColor.r - color.r, 2) +
                    Math.pow(testColor.g - color.g, 2) +
                    Math.pow(testColor.b - color.b, 2)
                )
            }));

            // Ordenar y seleccionar k vecinos más cercanos
            distances.sort((a, b) => a.distance - b.distance);
            const nearestNeighbors = distances.slice(0, k);

            // Contar frecuencia
            const frequency = {};
            nearestNeighbors.forEach(neighbor => {
                frequency[neighbor.nombre] = (frequency[neighbor.nombre] || 0) + 1;
            });

            // Predicción basada en frecuencia
            let predictedColor = null;
            let maxCount = 0;
            for (const [color, count] of Object.entries(frequency)) {
                if (count > maxCount) {
                    maxCount = count;
                    predictedColor = color;
                }
            }

            // Verificar si la predicción es correcta
            if (predictedColor === testColor.nombre) {
                correctPredictions++;
            }
        });

        // Calcular precisión
        const accuracy = (correctPredictions / testDataset.length) * 100;
        results.push({ k, accuracy });
    }

    plotResults(results);
}

function plotResults(results) {
    const canvas = document.getElementById("accuracy-chart");
    const ctx = canvas.getContext("2d");

    const labels = results.map(result => result.k);
    const data = results.map(result => result.accuracy);

    new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [{
                label: "Precisión (%)",
                data: data,
                borderColor: "blue",
                borderWidth: 2,
                fill: false
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: {
                    title: { display: true, text: "Valor de k" }
                },
                y: {
                    title: { display: true, text: "Precisión (%)" },
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });
}

const results = evaluateK(10);
plotResults(results);
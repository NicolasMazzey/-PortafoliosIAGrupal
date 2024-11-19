
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


// function predictColor() {
//     const r = parseInt(document.getElementById("r").value);
//     const g = parseInt(document.getElementById("g").value);
//     const b = parseInt(document.getElementById("b").value);

//     // Asegurarse de que r, g, b están dentro del rango esperado
//     if (isNaN(r) || isNaN(g) || isNaN(b) || r < 0 || r > 255 || g < 0 || g > 255 || b < 0 || b > 255) {
//         alert("Por favor, ingresa valores RGB válidos entre 0 y 255.");
//         return;
//     }

//     // Calcular el color más cercano (distancia euclidiana)
//     let closestColor = null;
//     let minDistance = Infinity;

//     dataset.forEach(color => {
//         const distance = Math.sqrt(
//             Math.pow(r - color.r, 2) +
//             Math.pow(g - color.g, 2) +
//             Math.pow(b - color.b, 2)
//         );


//         if (distance < minDistance) {
//             minDistance = distance;
//             closestColor = color.nombre;
//         }
//     });

//     document.getElementById("result").innerText = `Resultado: ${closestColor || 'No se encontró un color cercano'}`;
//     document.getElementById("color-box").style.backgroundColor = `rgb(${r}, ${g}, ${b})`;
// }

//Codigo del KNN

function predictColor(k = 3) { // k es el número de vecinos, por defecto 3
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

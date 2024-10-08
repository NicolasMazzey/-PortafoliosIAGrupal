// home.js
/*
// Seleccionando el contenedor de los proyectos
const projectGrid = document.getElementById('project-grid');

// Función para cargar los proyectos desde archivos.json
function loadProjects() {
    fetch('../assets/archivos.json')
        .then(response => {
            if (!response.ok) {
                throw new Error('Error al cargar el archivo JSON');
            }
            return response.json();
        })
        .then(projects => {
            projectGrid.innerHTML = '';

            projects.forEach(project => {
                const projectCard = document.createElement('div');
                projectCard.className = 'project-card';
                projectCard.dataset.id = project.id; // Añadimos un ID para identificar el proyecto

                const projectImg = document.createElement('img');
                projectImg.src = project.imagenes[0];
                projectImg.alt = project.titulo;

                const projectTitleElem = document.createElement('h3');
                projectTitleElem.textContent = project.titulo;

                projectCard.appendChild(projectImg);
                projectCard.appendChild(projectTitleElem);

                // Añadir el evento de clic a la card
                projectCard.addEventListener('click', () => {
                    window.open(`../Proyecto/project-details.html?id=${project.id}`, '_self');
                });

                projectGrid.appendChild(projectCard);
            });
        })
        .catch(error => {
            console.error('Error:', error);
        });
}

// Llamar a la función para cargar los proyectos al cargar la página
loadProjects();
*/

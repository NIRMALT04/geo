// Create the scene
const scene = new THREE.Scene();

// Create a camera
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.z = 5;

// Create a renderer
const renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// Create a wireframe globe
const globeGeometry = new THREE.SphereGeometry(5, 32, 32);
const wireframeMaterial = new THREE.LineBasicMaterial({ color: 0xffffff });
const wireframe = new THREE.LineSegments(new THREE.EdgesGeometry(globeGeometry), wireframeMaterial);
scene.add(wireframe);

// Load GeoJSON file
fetch('custom.geo.json')
    .then(response => response.json())
    .then(data => {
        data.features.forEach(feature => {
            const coordinates = feature.geometry.coordinates[0]; // Adjust based on your GeoJSON structure
            const points = coordinates.map(coord => {
                const lat = coord[1] * (Math.PI / 180); // Convert latitude to radians
                const lon = coord[0] * (Math.PI / 180); // Convert longitude to radians
                const x = 5 * Math.cos(lat) * Math.cos(lon);
                const y = 5 * Math.sin(lat);
                const z = 5 * Math.cos(lat) * Math.sin(lon);
                return new THREE.Vector3(x, y, z);
            });

            const continentGeometry = new THREE.BufferGeometry().setFromPoints(points);
            const continentLine = new THREE.Line(continentGeometry, new THREE.LineBasicMaterial({ color: 0xff0000, linewidth: 2 }));
            scene.add(continentLine);
        });
    })
    .catch(error => console.error('Error loading GeoJSON:', error));

// Add OrbitControls for interactive rotation
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true; // an animation loop is required when either damping or auto-rotation are enabled
controls.dampingFactor = 0.25;
controls.screenSpacePanning = false;
controls.minDistance = 5;
controls.maxDistance = 100;
controls.maxPolarAngle = Math.PI; // Limit vertical rotation

// Update controls in the animation loop
controls.update();

// Adjust camera position for better visibility
camera.position.z = 10;

// Render the scene with rotation
function animate() {
    requestAnimationFrame(animate);
    // wireframe.rotation.y += 0.005; // Comment out this line to stop rotation
    renderer.render(scene, camera);
}
animate();

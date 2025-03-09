
const baseURL = window.API_CONFIG.API_BASE_URL;

document.addEventListener('DOMContentLoaded', function () {

    const createModalEl = document.getElementById('create-modal');
    const createModal = new bootstrap.Modal(createModalEl, { keyboard: true });

    const updateModalEl = document.getElementById('update-modal');
    const updateModal = new bootstrap.Modal(updateModalEl, { keyboard: true });

    // Bind to open the create pop-up button
    document.getElementById('open-create-modal').addEventListener('click', function () {
        createModal.show();
    });

    // Binding create form submission event
    document.getElementById('create-form').addEventListener('submit', function (e) {
        e.preventDefault();
        createParkingLot();
    });

    // Binding update form submission event
    document.getElementById('update-form').addEventListener('submit', function (e) {
        e.preventDefault();
        updateParkingLot();
    });

    // Bind Refresh Button
    document.getElementById('refresh-btn').addEventListener('click', fetchAllParkingLots);

    // Initial load list
    fetchAllParkingLots();

    // Global mount for direct call in HTML
    window.editParkingLot = function (lot_id) {
        fetch(baseURL + '/parking_lot/' + lot_id)
            .then(res => res.json())
            .then(data => {
                if (data.message === 'Parking lot not found') {
                    alert(data.message);
                } else {
                    // Populate the fields of the update form
                    document.getElementById('update-id').value = data.lot_id;
                    document.getElementById('update-name').value = data.name;
                    document.getElementById('update-address').value = data.address;
                    document.getElementById('update-lat').value = data.latitude;
                    document.getElementById('update-lng').value = data.longitude;
                    document.getElementById('update-total').value = data.total_spaces;
                    document.getElementById('update-available').value = data.available_spaces;

                    // Show update pop-up window
                    updateModal.show();
                }
            })
            .catch(err => console.error('Edit lot error:', err));
    }

    window.deleteParkingLot = function (lot_id) {
        if (!confirm('Are you sure you want to delete this record?')) return;

        fetch(baseURL + '/parking_lot/' + lot_id, { method: 'DELETE' })
            .then(res => res.json())
            .then(data => {
                alert(data.message);
                fetchAllParkingLots();
            })
            .catch(err => console.error('Delete lot error:', err));
    }

    window.createParkingLot = function () {
        const name = document.getElementById('create-name').value;
        const address = document.getElementById('create-address').value;
        const latitude = parseFloat(document.getElementById('create-lat').value);
        const longitude = parseFloat(document.getElementById('create-lng').value);
        const total_spaces = parseInt(document.getElementById('create-total').value);
        const available_spaces = parseInt(document.getElementById('create-available').value);

        fetch(baseURL + '/parking_lot', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name,
                address,
                latitude,
                longitude,
                total_spaces,
                available_spaces
            })
        })
            .then(res => res.json())
            .then(data => {
                alert(data.message + ' (lot_id=' + data.lot_id + ')');
                document.getElementById('create-form').reset();
                createModal.hide();
                fetchAllParkingLots();
            })
            .catch(err => console.error('Create lot error:', err));
    }

    window.updateParkingLot = function () {
        const lot_id = document.getElementById('update-id').value;
        const name = document.getElementById('update-name').value;
        const address = document.getElementById('update-address').value;
        const latitude = parseFloat(document.getElementById('update-lat').value);
        const longitude = parseFloat(document.getElementById('update-lng').value);
        const total_spaces = parseInt(document.getElementById('update-total').value);
        const available_spaces = parseInt(document.getElementById('update-available').value);

        fetch(baseURL + '/parking_lot/' + lot_id, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                name,
                address,
                latitude,
                longitude,
                total_spaces,
                available_spaces
            })
        })
            .then(res => res.json())
            .then(data => {
                alert(data.message);
                updateModal.hide();
                fetchAllParkingLots();
            })
            .catch(err => console.error('Update lot error:', err));
    }

    function fetchAllParkingLots() {
        fetch(baseURL + '/parking_lots')
            .then(res => res.json())
            .then(data => {
                populateTable(data);
            })
            .catch(err => console.error('Fetch all lots error:', err));
    }

    function populateTable(data) {
        const tableBody = document.querySelector('#parking-lots-table tbody');
        tableBody.innerHTML = '';
        data.forEach(item => {
            const row = document.createElement('tr');
            row.innerHTML = `
        <td>${item.lot_id}</td>
        <td>${item.name}</td>
        <td>${item.address}</td>
        <td>${item.latitude}</td>
        <td>${item.longitude}</td>
        <td>${item.total_spaces}</td>
        <td>${item.available_spaces}</td>
        <td>${item.created_at}</td>
        <td>${item.updated_at}</td>
        <td>
          <button class="btn btn-sm btn-primary" onclick="editParkingLot(${item.lot_id})">Edit</button>
          <button class="btn btn-sm btn-danger" onclick="deleteParkingLot(${item.lot_id})">Delete</button>
        </td>
      `;
            tableBody.appendChild(row);
        });
    }
});

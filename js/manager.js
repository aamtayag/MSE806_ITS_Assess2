const baseURL = window.API_CONFIG.API_BASE_URL;

document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM fully loaded');

    // init
    const modalCreateParkingLot = new bootstrap.Modal(document.getElementById('modal-create-parkinglot'), { keyboard: true });
    const modalUpdateParkingLot = new bootstrap.Modal(document.getElementById('modal-update-parkinglot'), { keyboard: true });
    const modalCreatePrediction = new bootstrap.Modal(document.getElementById('modal-create-prediction'), { keyboard: true });
    const modalUpdatePrediction = new bootstrap.Modal(document.getElementById('modal-update-prediction'), { keyboard: true });

    // bind Parking Lot Related Events
    document.getElementById('btn-create-parkinglot').addEventListener('click', () => {
        document.getElementById('form-create-parkinglot').reset();
        modalCreateParkingLot.show();
    });
    document.getElementById('btn-refresh-parkinglots').addEventListener('click', () => {
        fetchAllParkingLots();
    });
    document.getElementById('form-create-parkinglot').addEventListener('submit', (e) => {
        e.preventDefault();
        createParkingLot();
    });
    document.getElementById('form-update-parkinglot').addEventListener('submit', (e) => {
        e.preventDefault();
        updateParkingLot();
    });

    // bind Prediction Related Events
    document.getElementById('btn-create-prediction').addEventListener('click', () => {
        document.getElementById('form-create-prediction').reset();
        modalCreatePrediction.show();
    });
    document.getElementById('btn-refresh-predictions').addEventListener('click', () => {
        fetchAllPredictions();
    });
    document.getElementById('btn-save-prediction').addEventListener('click', () => {
        savePrediction();
    });
    document.getElementById('btn-update-prediction').addEventListener('click', () => {
        updatePrediction();
    });

    // Tab Load data when switching
    document.querySelectorAll('#managementTabs .nav-link').forEach(tab => {
        tab.addEventListener('shown.bs.tab', (e) => {
            const target = e.target.getAttribute('aria-controls');
            if (target === 'content-parkinglots') {
                fetchAllParkingLots();
            } else if (target === 'content-models') {
                fetchAllModels();
            } else if (target === 'content-predictions') {
                fetchAllPredictions();
            }
        });
    });

    // -------------------------------
    // Parking Lot
    // -------------------------------
    window.editParkingLot = (lotId) => {
        fetch(`${baseURL}/parking_lot/${lotId}`)
            .then(res => res.json())
            .then(data => {
                if (data.message === 'Parking lot not found') {
                    alert(data.message);
                } else {
                    document.getElementById('input-update-id').value = data.lot_id;
                    document.getElementById('input-update-name').value = data.name;
                    document.getElementById('input-update-address').value = data.address;
                    document.getElementById('input-update-lat').value = data.latitude;
                    document.getElementById('input-update-lng').value = data.longitude;
                    document.getElementById('input-update-total').value = data.total_spaces;
                    document.getElementById('input-update-available').value = data.available_spaces;
                    modalUpdateParkingLot.show();
                }
            })
            .catch(err => console.error('Edit Parking Lot error:', err));
    };

    window.deleteParkingLot = (lotId) => {
        if (!confirm('Are you sure you want to delete this record?')) return;
        fetch(`${baseURL}/parking_lot/${lotId}`, { method: 'DELETE' })
            .then(res => res.json())
            .then(data => {
                alert(data.message);
                fetchAllParkingLots();
            })
            .catch(err => console.error('Delete Parking Lot error:', err));
    };

    const createParkingLot = () => {
        const name = document.getElementById('input-create-name').value;
        const address = document.getElementById('input-create-address').value;
        const latitude = parseFloat(document.getElementById('input-create-lat').value);
        const longitude = parseFloat(document.getElementById('input-create-lng').value);
        const totalSpaces = parseInt(document.getElementById('input-create-total').value);
        const availableSpaces = parseInt(document.getElementById('input-create-available').value);

        fetch(`${baseURL}/parking_lot`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, address, latitude, longitude, total_spaces: totalSpaces, available_spaces: availableSpaces })
        })
            .then(res => res.json())
            .then(data => {
                alert(`${data.message} (lot_id=${data.lot_id})`);
                modalCreateParkingLot.hide();
                fetchAllParkingLots();
            })
            .catch(err => console.error('Create Parking Lot error:', err));
    };

    const updateParkingLot = () => {
        const lotId = document.getElementById('input-update-id').value;
        const name = document.getElementById('input-update-name').value;
        const address = document.getElementById('input-update-address').value;
        const latitude = parseFloat(document.getElementById('input-update-lat').value);
        const longitude = parseFloat(document.getElementById('input-update-lng').value);
        const totalSpaces = parseInt(document.getElementById('input-update-total').value);
        const availableSpaces = parseInt(document.getElementById('input-update-available').value);

        fetch(`${baseURL}/parking_lot/${lotId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, address, latitude, longitude, total_spaces: totalSpaces, available_spaces: availableSpaces })
        })
            .then(res => res.json())
            .then(data => {
                alert(data.message);
                modalUpdateParkingLot.hide();
                fetchAllParkingLots();
            })
            .catch(err => console.error('Update Parking Lot error:', err));
    };

    const fetchAllParkingLots = () => {
        fetch(`${baseURL}/parking_lots`)
            .then(res => res.json())
            .then(data => {
                populateTable('table-parkinglots', data, renderParkingLotRow);
            })
            .catch(err => console.error('Fetch Parking Lots error:', err));
    };

    const renderParkingLotRow = (item) => {
        return `
      <tr>
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
      </tr>
    `;
    };

    // -------------------------------
    // Model Management
    // -------------------------------
    const fetchAllModels = () => {
        fetch(`${baseURL}/models`)
            .then(res => res.json())
            .then(data => {
                populateTable('table-models', data, renderModelRow);
            })
            .catch(err => console.error('Fetch Models error:', err));
    };

    const renderModelRow = (item) => {
        return `
      <tr>
        <td>${item.model_id}</td>
        <td>${item.model_name}</td>
        <td>${item.description}</td>
        <td>${item.created_at}</td>
        <td>${item.updated_at}</td>
        <td>
          <button class="btn btn-sm btn-primary" onclick="editModel(${item.model_id})">Edit</button>
          <button class="btn btn-sm btn-danger" onclick="deleteModel(${item.model_id})">Delete</button>
        </td>
      </tr>
    `;
    };

    // -------------------------------
    // Prediction Management
    // -------------------------------
    const fetchAllPredictions = () => {
        fetch(`${baseURL}/predictions`)
            .then(res => res.json())
            .then(data => {
                populateTable('table-predictions', data, renderPredictionRow);
            })
            .catch(err => console.error('Fetch Predictions error:', err));
    };

    const renderPredictionRow = (item) => {
        return `
      <tr>
        <td>${item.prediction_id}</td>
        <td>${item.parking_lot_id}</td>
        <td>${item.prediction_description}</td>
        <td>${item.prediction_datetime}</td>
        <td>${item.model || 'N/A'}</td>
        <td>${item.predicted_value || 'N/A'}</td>
        <td>${item.created_at}</td>
        <td>
          <button class="btn btn-sm btn-primary" onclick="editPrediction(${item.prediction_id})">Edit</button>
          <button class="btn btn-sm btn-danger" onclick="deletePrediction(${item.prediction_id})">Delete</button>
        </td>
      </tr>
    `;
    };

    // Generic table-filling function
    const populateTable = (tableId, data, renderRow) => {
        const tbody = document.getElementById(tableId).querySelector('tbody');
        tbody.innerHTML = data.map(renderRow).join('');
    };

    const savePrediction = () => {
        let dt = document.getElementById('input-create-prediction-time').value;
        dt = dt.replace('T', ' ') + ":00";
        const predictionData = {
            parking_lot_id: document.getElementById('input-create-lot-id').value,
            prediction_description: document.getElementById('input-create-prediction-description').value,
            prediction_datetime: dt,
            predicted_value: document.getElementById('input-create-predicted-value').value || null,
            model: document.getElementById('input-create-model').value || null
        };
        fetch(`${baseURL}/prediction`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(predictionData)
        })
            .then(res => res.json())
            .then(data => {
                alert(data.message);
                modalCreatePrediction.hide();
                fetchAllPredictions();
            })
            .catch(err => console.error('Create Prediction error:', err));
    };

    window.editPrediction = (predictionId) => {
        console.log(`editPrediction called with ID: ${predictionId}`);
        fetch(`${baseURL}/prediction/${predictionId}`)
            .then(res => res.json())
            .then(data => {
                if (data.message === 'Prediction not found') {
                    alert(data.message);
                } else {
                    document.getElementById('input-update-prediction-id').value = data.prediction_id;
                    document.getElementById('input-update-lot-id').value = data.parking_lot_id;
                    // Replace the space with "T" to fit the datetime-local format
                    document.getElementById('input-update-prediction-time').value = data.prediction_datetime.replace(' ', 'T');
                    document.getElementById('input-update-prediction-description').value = data.prediction_description || '';
                    document.getElementById('input-update-predicted-value').value = data.predicted_value || '';
                    document.getElementById('input-update-model').value = data.model || '';
                    modalUpdatePrediction.show();
                }
            })
            .catch(err => console.error('Edit Prediction error:', err));
    };

    const updatePrediction = () => {
        const predictionId = document.getElementById('input-update-prediction-id').value;
        const predictionData = {
            parking_lot_id: document.getElementById('input-update-lot-id').value,
            prediction_datetime: document.getElementById('input-update-prediction-time').value,
            prediction_description: document.getElementById('input-update-prediction-description').value,
            predicted_value: document.getElementById('input-update-predicted-value').value || null,
            model: document.getElementById('input-update-model').value || null
        };

        fetch(`${baseURL}/prediction/${predictionId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(predictionData)
        })
            .then(res => res.json())
            .then(data => {
                alert(data.message);
                modalUpdatePrediction.hide();
                fetchAllPredictions();
            })
            .catch(err => console.error('Update Prediction error:', err));
    };

    window.deletePrediction = (predictionId) => {
        if (!confirm('Are you sure you want to delete this prediction?')) return;
        fetch(`${baseURL}/prediction/${predictionId}`, { method: 'DELETE' })
            .then(res => res.json())
            .then(data => {
                alert(data.message);
                fetchAllPredictions();
            })
            .catch(err => console.error('Delete Prediction error:', err));
    };

    // Parking Lots data is loaded by default
    fetchAllParkingLots();
});

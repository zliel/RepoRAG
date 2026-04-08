// Sample JavaScript file for testing
class DataService {
    constructor() {
        this.data = [];
    }

    fetchData(url) {
        return fetch(url)
            .then(response => response.json());
    }

    processData(data) {
        return data.map(item => ({
            ...item,
            processed: true
        }));
    }
}

function initialize() {
    const service = new DataService();
    return service.fetchData('/api/data')
        .then(service.processData);
}

const config = {
    apiUrl: 'https://api.example.com',
    timeout: 5000
};

export { DataService, initialize, config };

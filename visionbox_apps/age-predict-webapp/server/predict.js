const axios = require('axios')

const predictCtonroller = {}

predictCtonroller.predictAge = (req, res, next) => {
    

    const imageData = {
        image: req.body.image
    }

    axios.post('http://localhost:9000/predict', imageData, axiosConfig)
        .then(response => {
            console.log(response.data)
            res.locals.data = response.data
        })
        .catch((err) =>console.error(err))
}

module.exports = predictCtonroller
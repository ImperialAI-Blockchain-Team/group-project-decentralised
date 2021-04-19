import web3 from './web3';
import { datasetDatabase} from "../abi/abi";

const address = '0xe1213Ae4386757239c90d30Ce6Ff5f5000BAD485';

export default new web3.eth.Contract(datasetDatabase, address);
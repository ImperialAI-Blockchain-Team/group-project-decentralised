import web3 from './web3';
import { datasetDatabase} from "../abi/abi";

const address = '0x34B54A8dA2c2f8Bb5cE8a9C52D56a7A591eB78b6';

export default new web3.eth.Contract(datasetDatabase, address);
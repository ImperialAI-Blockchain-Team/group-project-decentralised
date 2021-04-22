import web3 from './web3';
import { datasetDatabase} from "../abi/abi";

const address = '0xD34C8BF74A1f71f01c978bB34bEe50eaf4F0DD67';

export default new web3.eth.Contract(datasetDatabase, address);

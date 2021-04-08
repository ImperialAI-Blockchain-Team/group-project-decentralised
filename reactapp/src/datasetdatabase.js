import web3 from './web3';
import { datasetDatabase} from "./abi/abi";

const address = '0x61d9Bf9c81F9A7C9c19a916ca575f6F7B84627b6';

export default new web3.eth.Contract(datasetDatabase, address);
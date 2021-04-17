import web3 from './web3';
import { datasetDatabase} from "../abi/abi";

const address = '0xce83e15956a04AFd350b63de195fF3557052E6B1';

export default new web3.eth.Contract(datasetDatabase, address);
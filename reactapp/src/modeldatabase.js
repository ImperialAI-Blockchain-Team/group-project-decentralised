import web3 from './web3';
import { modelDatabase} from "./abi/abi";

const address = '0xedC94EF1A54dBD00729c64b151F48c493CdD7Ed5';

export default new web3.eth.Contract(modelDatabase, address);